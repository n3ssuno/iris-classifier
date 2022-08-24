#!/usr/bin/env python

"""
Tool to post-process the output of the classification phase.

For each page identified as a "true VPM page" the script checks 
 which patents, among the possible ones for that specific page,
 are actually present in the page and which not. 
It returns, for each entry of the database, a JSON line of the type
 {'vpm_page': 'URL_OF_THE_PAGE', 
  'is_true_vpm_page': true/false, 
  'is_patent_in_page': [(PATENT_NUMBER: true/false), 
                        (PATENT_NUMBER: true/false)]}

Author: Carlo Bottai
Copyright (c) 2021 - TU/e and EPFL
License: See the LICENSE file.
Date: 2021-05-08

"""


## LIBRARIES ##

import numpy as np
import pandas as pd
import os
import pathlib
from io import BytesIO
from hashlib import md5
from nltk.tokenize import sent_tokenize
import re
from urllib.parse import urlparse
from os.path import splitext
from datetime import datetime
import json

from flata import Flata, JSONStorage

from bs4 import BeautifulSoup as beautiful_soup
import html5lib

import pdfminer.high_level as pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import pdf2image
import pytesseract

from striprtf.striprtf import rtf_to_text

import asyncio
import aiofiles

from aiohttp import ClientSession, BadContentDispositionHeader

from tqdm.asyncio import tqdm as aio_tqdm
import warnings

from iris_utils.parse_args import parse_io


## TYPE HINTS ##

from typing import List, Tuple, Set, TypedDict
from pathlib import PosixPath
from flata.database import Table as fa_Table
class LineDict(TypedDict):
    db_id: int
    vpm_page: str
    patent_ids: int


## WARNINGS SUPPRESSION ##

# Suppress PDF text extraction not allowed warning 
#  and any other warning from the `pdfminer` module
warnings.filterwarnings('ignore', module = 'pdfminer')

# Suppress BadContentDispositionHeader warning 
#   from the `aiohttp` module
warnings.simplefilter('ignore', BadContentDispositionHeader)


#################
#   SETTINGS    #
#################

# Name of the filder where the local copy of the pages have been saved
files_folder = 'files'

# User agent
# Useful for both the type of documents (HTML and others) considered in the script
USER_AGENT = ('Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) '
              'Gecko/2009021910 Firefox/3.0.7')

# Length of the texts in the corpus of each document
# Number of characters, extracted from the full text of each document, 
#   before and after the keywords defined afterward
CONTEXT_SPAN = 500

# Choose the name of the log file where eventual errors will be reported
# The file will have a name like pre_classify_%Y_%m_%d_%H_%M.log
LOG_FILE = 'post_classify'

## ASYNCIO SETTINGS ##

# Run no more than 25 tasks at a time
NUM_CONCURRENT_TASKS = 25
SEMAPHORE = asyncio.Semaphore(NUM_CONCURRENT_TASKS)

################
#    REGEX     #
################

# Punctuation characters that will be removed
PUNCT_RE = re.compile(r'[\n\f\r\t\x0A\x0C\x0D\x09\s]+')

# Regular expressions used to convert a URL into a file name
HTTPWWW_RE = re.compile(r'^(.*:\/\/)?(www\.)?', flags = re.IGNORECASE)
NOALPHA_RE = re.compile(r'\W')

# Regular expression used to remove the sentences about cookie or privacy policy
# Useful to remove useless portions of the headers and footers
COOKIE_RE = re.compile(r'(cookie)|(privacy policy)', flags=re.IGNORECASE)

# Regular expression
PATNUM_RE = re.compile(r'\d{1,2},?\d{3},?\d{3}')


#################
#   FUNCTIONS   #
#################

def read_input(f_in: str) -> fa_Table:
    """
    Read the input file
    """
    
    DB = Flata(f_in, storage=JSONStorage)
    database = DB.table('iris_vpm_pages_classifier')
    
    return database

def generate_file_name(url: str) -> str:
    """ 
    Given the URL provided, return a standardized file name
    """

    # Remove 'https://', 'ftp://' and similar things, and remove 'www'
    file_name = HTTPWWW_RE.sub('', url)
    
    # Replace any non-alphanumeric chars with '_'
    file_name = NOALPHA_RE.sub('_', file_name)

    # If the generated filename is longer than 250 bytes
    #   (i.e., about the lenght-limit for an ext4 file system),
    #    then use as name an hash hexdigest string
    if len(file_name.encode()) >= 250:
        file_name = md5(file_name.encode()).hexdigest()

    return file_name

def which_content_type_exists(file_path: str) -> str:
    """
    Returns the content type based on which file exists locally
    Returns None if no file exists for the document of interest
    """
    for content_type in ['html', 'txt', 'rtf', 'pdf', 'other']:
        # NB PDF must always be the last one, since also HTML contents 
        #   have a PDF version (and potentialy other types 
        #   will do the same in the future)
        type_path = file_path.replace('.pdf', f'.{content_type}')
        if os.path.exists(type_path):
            return content_type.upper()
    return None

async def get_content_type(url: str, file_path: str, requests_session: ClientSession) -> str:
    """ 
    Determine the type of content returned by a GET request to the URL provided
    The possible answers are: 
      - HTML, PDF, TXT (documents handled by the script)
      - OTHER (documents unhandled by the script)
      - FAILED (generic error while connecting with the remote source)
    """

    local_content_type = which_content_type_exists(
        file_path = file_path)
    if local_content_type:
        return local_content_type

    # If the URL names a file that ends in *.pdf (*.txt) its a PDF (TXT)
    url_path = urlparse(url).path
    url_root, url_ext = splitext(url_path.lower())
    if url_ext.endswith('pdf'):
        return 'PDF'
    if url_ext.endswith('txt'):
        return 'TXT'
    
    try:
        # Require the HEAD for the URL
        response = await requests_session.request(
            method = 'HEAD', 
            url = url, 
            headers = {'User-Agent': USER_AGENT}, 
            allow_redirects = True, 
            ssl = False)
        
        # assert response.status in [200, 403]
        
        # Take the content-type from the HEAD
        remote_content_type = response.content_type

    except:
        return 'FAILED'

    # Is the content-type a PDF?
    if remote_content_type and remote_content_type.startswith('application/pdf'):
        return 'PDF'
    
    # Is the content-type an RTF?
    if remote_content_type and remote_content_type.startswith('application/rtf'):
        return 'RTF'
    
    # Is the content-type a plain text?
    if remote_content_type and remote_content_type.startswith('text/plain'):
        return 'TXT'

    # Is the content-type a stream of data?
    if remote_content_type and remote_content_type.startswith('application/octet-stream'):
        try:
            # Take the content-disposition from the HEAD
            content_disposition = response.content_disposition
            # Take the filename field from the content-disposition
            content_disposition = re.search(r'filename = "(.*)"', content_disposition)
        except:
            return 'FAILED'
        # Is the file a PDF?
        if content_disposition and \
           any([splitext(group.lower())[1].endswith('pdf') \
                for group in content_disposition.groups()]):
            return 'PDF'
        if content_disposition and \
           any([splitext(group.lower())[1].endswith('rtf') \
                for group in content_disposition.groups()]):
            return 'RTF'
        # Is the file a TXT?
        if content_disposition and \
           any([splitext(group.lower())[1].endswith('txt') \
                for group in content_disposition.groups()]):
            return 'TXT'
        # Is the file something else?
        else:
            return 'OTHER'
    
    # Is the content-type an HTML?
    if remote_content_type and remote_content_type.startswith('text/html'):
        return 'HTML'
    
    # Is the content-type something else?
    return 'OTHER'

async def get_content_from_url(url: str, requests_session: ClientSession) -> bytes:
    """ 
    Download the document from the URL provided, store it locally and return it 
    """

    try:
        # Download the content from the URL
        response = await requests_session.request(
            method = 'GET', 
            url = url, 
            headers = {'User-Agent': USER_AGENT}, 
            allow_redirects = True, 
            ssl = False)
        assert response.status == 200
    except:
        text_bytes = b''
    else:
        # Read the downloaded content
        try:
            text_bytes = await response.read()
        except:
            text_bytes = b''

    # Return the content
    return text_bytes

async def get_text_from_txt(url: str, file_path: str, requests_session: ClientSession) -> str:
    """
    Extract the text from the TXT file provided (or downloaded from the URL provided)
    """
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_in:
            text_bytes = f_in.read()
    else:
        text_bytes = await get_content_from_url(
            url = url, 
            requests_session = requests_session)
    text = text_bytes.decode(errors='ignore')
    
    return text

async def get_text_from_pdf(url: str, file_path: str, requests_session: ClientSession, use_ocr = False) -> str:
    """ 
    Extract the text from the PDF file provided (or downloaded from the URL provided)
    """

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_in:
            text_bytes = f_in.read()
    else:
        text_bytes = await get_content_from_url(
            url = url, 
            requests_session = requests_session)

    if use_ocr:
        try:
            pdf_parser = PDFParser(BytesIO(text_bytes))
            pdf = PDFDocument(pdf_parser)
            n_pages = pdf.catalog['Pages'].resolve()['Count']
            # Analyze the document only if it is shorter than 30 pages
            if n_pages<30:
                pages = pdf2image.convert_from_bytes(text_bytes, grayscale = True)
            
            text = ''
            for page in pages:
                page = pytesseract.image_to_string(page, lang = 'eng')
                text += page
        except:
            text = ''
    else:
        try:
            text = pdfminer.extract_text(BytesIO(text_bytes))
        except:
            text = ''
    
    return text

async def get_text_from_rtf(url: str, file_path: str, requests_session: ClientSession) -> str:
    """
    Extract the text from the RTF file provided (or downloaded from the URL provided)
    """

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_in:
            text_bytes = f_in.read()
    else:
        text_bytes = await get_content_from_url(
            url = url, 
            requests_session = requests_session)
    
    try:
        text = text_bytes.decode(errors='ignore')
        text = rtf_to_text(text)
    except:
        text = ''

    return text

async def get_text_from_html(url: str, file_path: str, requests_session: ClientSession) -> Tuple[str, List[str]]:
    """ 
    Extract the text from the body of the document, 
      using the local version of the website (or try to create one)
    """

    html_path = file_path.replace('.pdf', '.html')
    
    # Use the, previously stored, local HTML version of the URL, if exists
    try:
        if html_path and os.path.exists(html_path):
            with open(html_path, 'r') as f_in:
                html_soup = beautiful_soup(f_in, 'html5lib')
        else:
            text_bytes = await get_content_from_url(
                url = url,
                requests_session = requests_session)
            html_soup = beautiful_soup(text_bytes, 'html5lib')
    except:
        text = ''
    else:
        try:
            # Remove <script> and <style> tags
            script_style = [el.extract() \
                for tag in ['script','style'] \
                    for el in html_soup.find_all(tag)]
            
            # Select the <body> of the page
            body = html_soup.find('body')
            
            # # Remove hidden elements
            # hidden_elements = [el.extract() \
            #     for el in body.find_all(
            #         style=re.compile(f'display:\s*none'))]
            
            # Extract text from the <body>
            text = body.get_text(separator=' ')

        except:
            text = ''
    
    return text

async def get_text(vpm_page: str, vpm_page_path: str, requests_session: ClientSession) -> Tuple[str, str]:
    """ 
    Extract text from the URL provided
    Return both the document content and the document type
    Note: When relevant, use also the OCR
    """

    content_type = await get_content_type(
        url = vpm_page, 
        file_path = vpm_page_path,
        requests_session = requests_session)
    
    if content_type == 'PDF':
        text = await get_text_from_pdf(
            url = vpm_page, 
            file_path = vpm_page_path, 
            requests_session = requests_session)
    elif content_type == 'TXT':
        file_path = vpm_page_path.replace('.pdf', '.txt')
        text = await get_text_from_txt(
            url = vpm_page, 
            file_path = file_path,
            requests_session = requests_session)
    elif content_type == 'RTF':
        file_path = vpm_page_path.replace('.pdf', '.rtf')
        text = await get_text_from_rtf(
            url = vpm_page,
            file_path = file_path, 
            requests_session = requests_session)
    elif content_type == 'HTML':
        text = await get_text_from_html(
            url = vpm_page, 
            file_path = vpm_page_path, 
            requests_session = requests_session)
        # Remove sentences if 'cookie' or 'privacy policy' are named
        text = re.sub(' \.+', '', 
            ' '.join([sentence \
                for sentence in sent_tokenize(re.sub('(\n\s?){2,}', '. ', text)) \
                    if not (COOKIE_RE.search(sentence) and len(sentence)<CONTEXT_SPAN)]))
    else:
        text = ''

    # Remove new lines, tabs and spaces
    text = PUNCT_RE.sub(' ', text).strip()
    
    return text, content_type

async def search_patent(vpm_page_text: str, patent_ids: Set[int]) -> Set[int]:
    """
    Search any substring in the document text that matches 
      a pattern compatible with a patent number
    Return the intersection between the matched substrings 
      and the relevant patent numbers
    """

    patents_in_text = PATNUM_RE.findall(vpm_page_text)
    patents_in_text = [patent.replace(',', '') \
        for patent in patents_in_text  if not patent.startswith('0')]
    patents_in_text = [int(patent) for patent in patents_in_text]
    identified_patents = patent_ids.intersection(patents_in_text)

    return identified_patents

async def search_patent_and_write_results(db_id: int, vpm_page: str, patent_ids: Set[str], files_folder: str, requests_session: ClientSession, out_path: PosixPath) -> bool:
    """
    Search which of the relevant patent numbers appears in the text of the document 
      and write the results in the database
    """

    # Skip the line if there is no VPM page to check
    if not vpm_page:
        out_data = {'id': db_id, 'is_patent_in_page': None}
        
    else:
        vpm_page_path = re.sub(r'/$', '', vpm_page)
        vpm_page_path = f'{files_folder}/{generate_file_name(vpm_page_path)}.pdf'
        vpm_page_text, content_type = await get_text(
            vpm_page = vpm_page, 
            vpm_page_path = vpm_page_path, 
            requests_session = requests_session)

        identified_patents = await search_patent(
            vpm_page_text = vpm_page_text,
            patent_ids = patent_ids)
        
        # For PDF files, if no patent id has been identified, 
        #   use OCR on the document and try to extract the patent ids again
        if content_type == 'PDF' and len(identified_patents) == 0:
            vpm_page_text = await get_text_from_pdf(
                url = vpm_page, 
                file_path = vpm_page_path, 
                requests_session = requests_session, 
                use_ocr = True)
            identified_patents = await search_patent(
                vpm_page_text = vpm_page_text,
                patent_ids = patent_ids)

        is_patent_in_page = []
        for parent_id in patent_ids:
            is_in_page = parent_id in identified_patents
            is_patent_in_page.append((parent_id, is_in_page))

        out_data = {'id': db_id, 'is_patent_in_page': is_patent_in_page}
    
    out_data = json.dumps(out_data)
    
    async with aiofiles.open(out_path, 'a') as f_out:
        await f_out.write(out_data+'\n')

    return True


#################
#  PARALLELIZE  #
#################

async def run_task(line: LineDict, files_folder: str, requests_session: ClientSession, out_path: PosixPath) -> None:
    """ 
    Run the parser and writer asynchronously 
      (limiting to 25 the maximum number of tasks run at a time)
    """

    async with SEMAPHORE:
        results = await search_patent_and_write_results(
            db_id = line['id'],
            vpm_page = line['vpm_page'], 
            patent_ids = set(line['patent_id']), 
            files_folder = files_folder, 
            requests_session = requests_session, 
            out_path = out_path)
        return results


################
#     MAIN     #
################

async def main() -> None:
    """
    Read the input files, create the tasks and run them asynchronously
    """

    # Read the information from the terminal
    args = parse_io()

    # Present working directory
    pwd = pathlib.Path(__file__).parent

    in_path = pwd.joinpath(args.input)
    out_path = pwd.joinpath(args.output)

    # Read the input database
    database = read_input(in_path)

    # Extract only the useful information from the database
    data = [{k:v for k,v in line.items() if k in ['id', 'vpm_page', 'patent_id']} \
        for line in database]
    
    # If the output data file already exists, 
    #   remove the lines already analyzed from the data to check
    if os.path.exists(out_path):
        with open(out_path, 'r') as f_out:
            bak_ids = [json.loads(line) \
                for line in f_out.read().splitlines() if line!='']
            bak_ids = [int(line['id']) for line in bak_ids]
        
        data = [line for line in database if line['id'] not in bak_ids]

    # Remove from the data the lines with no VPM page to analyze 
    #   and write them in the output data file
    data_none = [{'id': line['id'], 'is_patent_in_page': None} \
        for line in data if not line['vpm_page']]
    data = [line for line in data if line['vpm_page']]
    if len(data_none):
        with open(out_path, 'a') as f_out:
            data_none = '\n'.join([json.dumps(line) for line in data_none])+'\n'
            data_none = re.sub('\n+', '\n', data_none)
            f_out.write(data_none)

    async with ClientSession() as requests_session:
        
        # Create a task for each line
        tasks = [asyncio.ensure_future(
            run_task(
                line = line,
                files_folder = files_folder,
                requests_session = requests_session,
                out_path = out_path)) \
                    for line in data]
        
        # Run the tasks
        for task in aio_tqdm(asyncio.as_completed(tasks), total = len(tasks)):
            task_result = await task
    
    # TODO This version of the script uses a temporary file (the one passed 
    #      as output file) to store the data and, at the end, asks the user 
    #      if she wants to update the input database accordingly with the 
    #      results stored in the temporary file. This is because the Flata 
    #      package has some issues with asynchronous writing. This is not the 
    #      best solution and it must be improved in a future version 
    #      of the script
    print('All the entries have been analyzed')
    update_database = input('Do you want to update the input database with the output of this script? [y/N] ')
    if update_database == 'y':
        print('Please, wait until all the results have been written in the input database')
        print('It can take some minute to complete this task')

        bak_path = args.input
        bak_path = bak_path.replace('.json', '')
        bak_path = datetime.now().strftime(f'{bak_path}_%Y_%m_%d_%H_%M.json')
        bak_path = pwd.joinpath(bak_path)
        DB_bak = Flata(bak_path, storage=JSONStorage)
        database_bak = DB_bak.table('iris_vpm_pages_classifier')
        
        data_db = [line for line in database.all()]
        data_db_ids = [line['id'] for line in data_db]
        
        with open(out_path, 'r') as f_tmp:
            data_toadd = [json.loads(line) \
                for line in f_tmp.read().splitlines() if line!='']
        data_toadd_ids = [line['id'] for line in data_toadd]

        added_data = database_bak.insert_multiple(data_db)
        print(f'A backup copy of the input database has been stored in {bak_path}')

        data_db = pd.DataFrame(
            data_db, index = data_db_ids) \
            .drop(columns = 'id') \
            .replace({None: np.nan})
        
        data_toadd = pd.DataFrame(
            data_toadd, index = data_toadd_ids) \
            .drop(columns = 'id') \
            .replace({None: np.nan})
        
        data_db.update(data_toadd)

        id_min = data_db.index.min()
        id_max = data_db.index.max()
        id_all = set(range(id_min, id_max))
        try:
            assert len(id_all.difference(data_db.index)) == 0
        except:
            print('Something went wrong while updating the database')
            print(f'The output of the analysis in anyhow store in {out_path}')
            print('Error: the following expected IDs are missing')
            print(id_all.difference(data_db.index))
        else:
            data_db = data_db \
                .replace({np.nan: None}) \
                .to_dict('records')
            database.purge()
            added_data = database.insert_multiple(data_db)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except Exception as e:
        log_file = datetime.now().strftime(f'{LOG_FILE}_%Y_%m_%d_%H_%M.log')
        print('Something went wrong.')
        print(f'Please, have a look at {log_file} for further details.')
        with open(log_file, 'w') as f_log:
            f_log.write(repr(e)+'\n')
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
