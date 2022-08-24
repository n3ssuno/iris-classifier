#!/usr/bin/env python

"""
Tool to pre-process the output of the scraping phase.

It automatically classifies those cases that are obviously 
(or better, very likely) either true positives or 
false positives into their respective categories.

Author: Carlo Bottai
Copyright (c) 2020 - TU/e and EPFL
License: See the LICENSE file.
Date: 2020-10-23

"""


## LIBRARIES ##

import numpy as np
import os
import json
import pathlib
from io import BytesIO
from PIL import Image
import warnings
from datetime import datetime
from hashlib import md5
import networkx 
from networkx.algorithms.components.connected import connected_components
from tqdm.asyncio import tqdm
from iris_utils.parse_args import parse_io

import re
from nltk.tokenize import sent_tokenize
from urllib.parse import urlparse, urljoin
from os.path import splitext

import asyncio
from asyncio.tasks import sleep
import aiofiles

from aiohttp import ClientSession, BadContentDispositionHeader

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup as beautiful_soup
import html5lib

import pdfminer.high_level as pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import pdf2image
import pytesseract

from striprtf.striprtf import rtf_to_text


## TYPE HINTS ##

from typing import List, TypedDict, Tuple, Generator, Pattern
from pathlib import PosixPath
from playwright.async_api._context_manager import \
    PlaywrightContextManager as ContextManager
from playwright.async_api._generated import \
    ChromiumBrowserContext as BrowserContext
class LineDict(TypedDict):
    urls: List[str]
    patent_ids_regex: Pattern
class DocInfoDict(TypedDict):
    url: str
    content_type: str
    headers: List[str]
    corpus: List[str]
    footers: List[str]
    text_in_imgs: List[str]
    patent_ids_regex: Pattern
    exclude_websites: List[str]


## WARNINGS SUPPRESSION ##

# Suppress PDF text extraction not allowed warning 
#  and any other warning from the `pdfminer` module
warnings.filterwarnings('ignore', module = 'pdfminer')

# Suppress BadContentDispositionHeader warning 
#   from the `aiohttp` module
warnings.simplefilter('ignore', BadContentDispositionHeader)

# Suppress DecompressionBombWarning warning 
#   from the `PIL` module
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


#################
#   SETTINGS    #
#################

# Set DEBUGGING to True and choose the SAMPLE_SIZE you like 
#   to analyze only a portion of the data provided
DEBUGGING = False
SAMPLE_SIZE = 100

# Choose the name of the log file where eventual errors will be reported
# The file will have a name like pre_classify_%Y_%m_%d_%H_%M.log
LOG_FILE = 'pre_classify'

# User agent
# Useful for both the type of documents (HTML and others) considered in the script
USER_AGENT = ('Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) '
              'Gecko/2009021910 Firefox/3.0.7')

# Label of the URL in the output file
URL_LABEL = 'VPM_PAGE'
PATENT_ID_LABEL = 'PATENT_ID'

# Length of the texts in the corpus of each document
# Number of characters, extracted from the full text of each document, 
#   before and after the keywords defined afterward
# Notes:
#  - The header of each document is defined as the first portion of its 
#      full text.
#  - The footer of each document is defined as the last portion of its 
#      full text.
#  - The length of the each header or footer is equal to one fourth 
#      of the context span
CONTEXT_SPAN = 500

## ASYNCIO SETTINGS ##

# Run no more than 50 tasks at a time
NUM_CONCURRENT_TASKS = 50
SEMAPHORE = asyncio.Semaphore(NUM_CONCURRENT_TASKS)

## PLAYWRIGHT SETTINGS ##

# Use the browser in headless mode
# NB If you set it to False, the script cannot save a PDF version of the visited websites
HEADLESS = True

# Set the navigation timeout to 5min for the browser (text extraction from the HTML documents)
TIMEOUT = 300000

# Add a margin of 0.75in to the PDF version of the HTML documents
PDF_MARGIN = '0.75in'

# Use mobile version of the HTML documents?
USE_MOBILE = False

# Viewport and other settings of the browser context
CONTEXT = {
    'viewport': {
        'width': 1768,
        'height': 992},
    'device_scale_factor': 1,
    'is_mobile': USE_MOBILE,
    'has_touch': USE_MOBILE,
    'user_agent': USER_AGENT}

# Chromium browser configuration parameters
use_mobile = lambda mobile: "true" if mobile else "false"
BROWSER_CONFIG = [
  f'--use-mobile-user-agent={use_mobile(USE_MOBILE)}',
  f'--user-agent={USER_AGENT}',
  '--ignore-certificate-errors',
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',
  '--disable-accelerated-2d-canvas',
  '--disable-gpu',
  '--window-position=0,0',
  '--start-fullscreen',
  '--hide-scrollbars']


################
#    REGEX     #
################

# Punctuation characters that will be removed
PUNCT_RE = re.compile(r'[\n\f\r\t\x0A\x0C\x0D\x09\s]+')

# The corpus is created taking the 250 characters 
#   around each of the following regular expressions
CORPUS_RES = [
    re.compile(regex, flags = re.IGNORECASE) for regex in [
        r'patent',
        r'marking',
        r'(^|\W)U\.?S\.?C\.?(\W|$)',
        r'United States Code',
        r'America Invents Act',
        r'(^|\W)A\.?I\.?A\.?(\W|$)',
        r'Securities and Exchange Commission', 
        r'Form ([A-Z]|[0-9]{1,2})-([A-Z]|[0-9]{1,2})',
        r'(^|\W)404(\W|$)']]

# Regular expression used to remove the sentences about cookie or privacy policy
# Useful to remove useless portions of the headers and footers
COOKIE_RE = re.compile(r'(cookie)|(privacy policy)', flags=re.IGNORECASE)

# Regular expression used by the URL and LAW rules
PATMARK_RE = re.compile(r'(virtual|patents?).?marking', flags = re.IGNORECASE)

# Regular expressions used by the LAW rule
LAW_RES = [
    re.compile(regex, flags = re.IGNORECASE) for regex in [
        r'America Invents Act', 
        r'35 U\.?S\.?C\.?(\ssect)?\W*287',
        r'287\(a\) of Title 35 of the United States Code']]

# Regular expressions used by the TEXT rule
TEXTP_RE = re.compile(r'patent', flags = re.IGNORECASE)
TEXT1_RE = re.compile(
    r'(^|\s)(protect|cover)[a-z]* (by|under|our)', 
    flags = re.IGNORECASE)
TEXT2_RE = re.compile(r'(^|\s)manufactur[a-z]* under', flags = re.IGNORECASE)
TEXT3_RE = re.compile(r'patent\W*protected', flags = re.IGNORECASE)
TEXT4_RE = re.compile(r'our patented', flags = re.IGNORECASE)
TEXT5_RE = re.compile(
    r'(^|\s)(((emplo|appl(y|ie))[a-z]*|uses?) (the|a|our|a number|several|some)|using our) .{0,50}patent', 
    flags = re.IGNORECASE)

# Symbols used by the TRADEMARK rule
R_TM_RES = [
    re.compile(regex, flags = re.IGNORECASE) for regex in [
        r'®', 
        r'\(r\)', 
        r'™']]

# Regular expressions used by the COPYRIGHT rule
C_RES = [
    re.compile(regex, flags = re.IGNORECASE) for regex in [
        r'©\W?[0-9]{4}', 
        r'[0-9]{4}\W?©', 
        r'\(c\)\W?[0-9]{4}', 
        r'[0-9]{4}\W?\(c\)', 
        r'Copyright\W?[0-9]{4}',
        r'[0-9]{4}\W?Copyright']]

# Regular expressions used by the SEC rule
# The second rule is a generalization of cases like "Form 10-Q", "Form 10-K", "Form S-1"
SEC_RE = re.compile(
    (r'United\W*States\W*Securities\W*and\W*Exchange\W*Commission\W*Washington'
     r'\W*D\W?C\W*[0-9]+\W*(Form|Schedule)\W*([A-Z]|[0-9]{1,2})-?([A-Z]|[0-9]{1,2})'),
    flags = re.IGNORECASE)

# Text used by the PATENT rule
# It matches "United States Patent" with title cases only and 
#   without the "s" after "Patent" to exclude cases of lists 
#   of patents preceded by the headline "United States Patents:"
PAT_RES = [
    re.compile(regex) for regex in [
        r'United States Patent([^s].*|$)',
        r'United States.*Patent Application Publication',
        (r'The Director of the United States.*'
         r'Patent and Trademark Office.*'
         r'Has received an application for a patent')]]

NF_RES = [
    re.compile(regex) for regex in [
        r'Error\W?404',
        r'404\W*((File\W)(or\WDirectory\W)?)?(Page\W)?Not\WFound']]

# Regular expressions used to convert a URL into a file name
HTTPWWW_RE = re.compile(r'^(.*:\/\/)?(www\.)?', flags = re.IGNORECASE)
NOALPHA_RE = re.compile(r'\W')


################
#    RULES     #
################

async def patent_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Is the document the PDF of the patent itself? 
    If you can find the sentence `United States Patent` in the first 250 characters 
      of the document, this is considered the PDF of the patent itself
    Notes:
      - Works only for the PDF documents
      - The number of characters included (250 by default) depends on the ones chosen in the 
          get_corpus() function
    """
    
    headers = doc_info['headers']
    content_type = doc_info['content_type']

    if content_type == 'PDF':
        for header in headers:
            if any([PAT_RE.search(header) for PAT_RE in PAT_RES]):
                return True
    return False

async def excluded_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Is the URL provided included in the list of websites considered irrelevant?
    Note: In theory, you have already excluded these pages during the scraping process.
      However, you can add other domains to the list afterward to help the classifier.
      Moreover, it is possible that the scraper did find a page also in one 
      of the excluded domains since the bar.foo.com is anyhow explored by Google 
      if you do not exclude foo.com completely
    """

    url = doc_info['url']
    exclude_websites = doc_info['exclude_websites']

    url = re.sub(r'.*://(www.)?', '', url)
    for exclude_website in exclude_websites:
        exclude_website_len = len(exclude_website.split('.'))
        url_check = '.'.join(url.split('/')[0].split('.')[-exclude_website_len:])
        if url_check.startswith(exclude_website):
            return True
    return False

async def url_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Does the URL provided contain the expression `virtual (patent) marking`?
    """
    
    url = doc_info['url']
    
    return PATMARK_RE.search(url) is not None

async def law_rule(doc_info: DocInfoDict) -> bool:
    """
    Is the America Invents Act named in the corpus of the document provided?
    NB What you can find here depends on the rules used to extract the corpus from the full text
    """
    
    corpus = doc_info['corpus']
    
    for text in corpus:
        if any([LAW_RE.search(text) for LAW_RE in LAW_RES+[PATMARK_RE]]):
            return True
    return False

async def text_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Is any of the relevant regexs present in the corpus of the document provided?
    NB What you can find here depends on the rules used to extract the corpus from the full text
    """
    
    corpus = doc_info['corpus']
    PATNUM_RE = doc_info['patent_ids_regex']

    for text in corpus:
        if ((TEXT1_RE.search(text) or TEXT2_RE.search(text)) and \
            (TEXTP_RE.search(text) or PATNUM_RE.search(text))) or \
           TEXT3_RE.search(text) or \
           TEXT4_RE.search(text) or \
           TEXT5_RE.search(text):
            return True
    return False

async def sec_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Is the document provided a document required by the Securities and Exchange Commission?
    """
    
    corpus = doc_info['corpus']
    
    for text in corpus: 
        if SEC_RE.search(text):
            return True
    return False

async def trademark_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Does the corpus extracted by the document provided contain any trademark symbol
      close to one of the relevant patent numbers?
    """
    
    corpus = doc_info['corpus']
    PATNUM_RE = doc_info['patent_ids_regex']

    for text in corpus:
        if any([R_TM_RE.search(text) for R_TM_RE in R_TM_RES]) and PATNUM_RE.search(text):
            return True
    return False

async def copyright_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Does the last element of the corpus extracted by the document provided contain any copyright symbol
      close to one of the relevant patent numbers?
    """
    
    footers = doc_info['footers']
    PATNUM_RE = doc_info['patent_ids_regex']

    for footer in footers:
        if any([C_RE.search(footer) for C_RE in C_RES]) and PATNUM_RE.search(footer):
            return True
    return False

async def img_rule(doc_info: DocInfoDict) -> bool:
    """ 
    Does any of the relevant patent numbers appear in one of the images included in the document provided?
    Note: Works only for the HTML documents (provided that the OCR is not used on them)
    """
    
    text_in_imgs = doc_info['text_in_imgs']
    PATNUM_RE = doc_info['patent_ids_regex']

    for text in text_in_imgs:
        if PATNUM_RE.search(text):
            return True
    return False

async def nocorpus_rule(doc_info: DocInfoDict) -> bool:
    """
    Does none of the relevant keywords has been found in the corpus, 
      but there is something in the header (meaning that the document has been read by the script)?
    Note: It is useful to combine it with the IMG rule 
      (since the patent number can be in one of the images of the document)
    """
    
    headers = doc_info['headers']
    corpus = doc_info['corpus']
    
    not_empty_header = len(headers)>0 and not all([len(header)==0 for header in headers])
    empty_corpus = len(corpus)==0 or all([len(text)==0 for text in corpus])
    
    return not_empty_header and empty_corpus

async def notfound_rule(doc_info: DocInfoDict) -> bool:
    """
    Was any `Error 404` statement found in the document?
    """

    # TODO This rule works very poorly because, in most cases, 
    #   it's written '404' without 'Error' in front or 'Not Found'
    #   See e.g.
    #   - https://www.eppendorf.com/uploads/media/Eppendorf_2021_AK01003931_-_FR_V1.pdf
    #   Another issue to solve is that, if there was a PDF file, 
    #     it will be expected a PDF file, but the document actually 
    #     found is an HTML page
    #   See e.g.
    #   - https://www.thermal-dynamics.com/thermal-dynamics/shared/content/requestliterature/de/upload/0-5120de-cutmaster-a40_a60-ac.pdf

    corpus = doc_info['corpus']

    for text in corpus:
        if any([NF_RE.search(text) for NF_RE in NF_RES]):
            return True
    return False

async def patnuminurl_rule(doc_info: DocInfoDict) -> bool:
    """
    Does any of the relevant patent numbers appear in the URL of the document provided?
    """

    url = doc_info['url']
    PATNUM_RE = doc_info['patent_ids_regex']

    if PATNUM_RE.search(url):
        return True
    return False

# Switch-case-like object
# Note: if you define a new rule, you can add it here 
#   and it will be applied automatically during the analysis
RULE_FS = {
    'EXCLUDED': excluded_rule, 
    'PATENT': patent_rule, 
    'SEC': sec_rule,
    'URL': url_rule, 
    'LAW': law_rule, 
    'TEXT': text_rule,
    'TRADEMARK': trademark_rule,
    'COPYRIGHT': copyright_rule,
    'IMG': img_rule,
    'NOCORPUS': nocorpus_rule,
    'NOTFOUND': notfound_rule,
    'PATNUMINURL': patnuminurl_rule}


#################
#   FUNCTIONS   #
#################

def generate_file_name(url: str, files_folder: str) -> str:
    """ 
    Given the URL provided, return a standardized file name
    """

    # Remove 'https://', 'ftp://' and similar things, and remove 'www'
    file_name = HTTPWWW_RE.sub('', url)
    
    # Replace any non-alphanumeric chars with '_'
    file_name = NOALPHA_RE.sub('_', file_name)

    # If the generated filename is longer than 250 bytes
    #   (i.e., about the lenght-limit for an ext4 file system),
    #    then use as name an hash hexdigest string and write the 
    #   corresponding URL in a file named <FILENAME>.url
    if len(file_name.encode()) >= 250:
        file_name = md5(file_name.encode()).hexdigest()
        with open(f'{files_folder}/{file_name}.url', 'w') as f_out:
            f_out.write(url)

    return file_name

def generate_patent_ids_regex(patent_ids: List[int]) -> Pattern:
    """ 
    Given that we know that at least one of the parent numbers in this list is present in the URL of interest, 
      convert the list of patent numbers into a convenient regex
    It takes into accout of patterns like 'US7194162'; '8,926,731'; '10 088 283'
    """

    patnum_re = [str(patent_id) for patent_id in patent_ids]
    patnum_re = [f'{patent_id[:-6]}.?{patent_id[-6:-3]}.?{patent_id[-3:]}' for patent_id in patnum_re]
    patnum_re = '|'.join(patnum_re)
    patnum_re = r'(^|[^0-9])({})([^0-9]|$)'.format(patnum_re)
    patnum_re = re.compile(patnum_re, re.IGNORECASE)

    return patnum_re

def deduplicate_data(data):
    """
    Look into the data and merge lines with at least one VPM page in common
    The new line will be composed of the union of the pages of the two (or more) joined lines and
      by the union of the patents relevant for one or the other line
    """

    def list_to_edges(vpm_pages):
        vpm_pages_iterator = iter(vpm_pages)
        next_page = next(vpm_pages_iterator)

        for current_page in vpm_pages_iterator:
            yield next_page, current_page
            next_page = current_page
    
    G = networkx.Graph()
    for line in data:
        G.add_nodes_from(line['vpm_pages'])
        edges_vpm_pages = list_to_edges(line['vpm_pages'])
        G.add_edges_from(edges_vpm_pages)
    
    data_connected_components = [vpm_page for vpm_page in connected_components(G)]
    
    data_dedup = []
    for line_connected_components in data_connected_components:
        patent_connected_components = []
        for line in data:
            vpm_pages = line['vpm_pages']
            if set(line_connected_components).intersection(vpm_pages):
                patent_connected_components.extend(line['patent_id'])
        line_dedup = {
            'vpm_pages': list(line_connected_components), 
            'patent_id': list(set(patent_connected_components))}
        data_dedup.append(line_dedup)
    
    return data_dedup

def which_content_type_exists(file_path: str) -> str:
    """
    Returns the content type based on which file exists locally
    Returns None if no file exists for the document of interest
    """
    for content_type in ['html', 'txt', 'rtf', 'other', 'pdf']:
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

async def get_content_from_url(url: str, file_path: str, requests_session: ClientSession) -> bytes:
    """ 
    Download the document from the URL provided, store it locally and return it 
    """

    # TODO
    # Check whether the request has been redirected and report also the actual URL
    # The EXCLUDED rule should look at both the URLs
    # See e.g.
    # - https://www.parc.com/patent/obtaining-spectral-information-from-moving-objects/

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
        # Write the downloaded content locally
        else:
            with open(file_path, 'wb') as f_out:
                f_out.write(text_bytes)

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
            file_path = file_path,
            requests_session = requests_session)
    text = text_bytes.decode(errors='ignore')
    
    return text

async def get_text_from_pdf(url: str, file_path: str, requests_session: ClientSession, use_ocr: bool = False) -> str:
    """ 
    Extract the text from the PDF file provided (or downloaded from the URL provided)
    Notes:
     - If use_ocr is True, transform the PDF in a PNG file and extract the text 
         from this last (looking into the first and last 5 pages only)
     - If the file already exists locally, use it instead of downloading it 
         from the remote source
    """

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_in:
            text_bytes = f_in.read()
    else:
        text_bytes = await get_content_from_url(
            url = url, 
            file_path = file_path, 
            requests_session = requests_session)

    if use_ocr:
        try:
            pdf_parser = PDFParser(BytesIO(text_bytes))
            pdf = PDFDocument(pdf_parser)
            n_pages = pdf.catalog['Pages'].resolve()['Count']
            if n_pages>10:
                pages = \
                    pdf2image.convert_from_bytes(text_bytes, grayscale = True, last_page = 5) + \
                    pdf2image.convert_from_bytes(text_bytes, grayscale = True, first_page = n_pages-4)
            else:
                pages = pdf2image.convert_from_bytes(text_bytes, grayscale = True)
            # Look into the first and the last 10 pages
            # This because it's likely that the useful information 
            #  will be in the front page (or in the first few page)
            #  or at the end of the document. Otherwise there are 
            #  very long documents that take an enormous amount of time 
            #  to be analyzed
            
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

async def get_text_from_imgs(url: str, url_imgs: List[str], requests_session: ClientSession) -> List[str]:
    """
    Use OCR to extract text components from any image present in a website 
      (provided that the image is store within the same domain of the main web site)
    """
    
    # TODO Consider using fuzzy matching with the NUMPAT_RE to extract also cases with one (or few) errors
    #      The 'regex' package replaces 're' and has also fuzzy matching

    url_base = urlparse(url).netloc
    
    is_abs_url = lambda url_img: \
        url_img.startswith('http://') or \
        url_img.startswith('www.') or \
        url_img.startswith('data:img/') or \
        url_img.find(url_base) >= 0
    url_imgs = [url_img if is_abs_url(url_img) else urljoin(url, url_img) \
        for url_img in url_imgs]
    url_imgs = [url_img \
        for url_img in url_imgs if url_img.find(url_base) >= 0]
    
    text_in_imgs = []
    for url_img in url_imgs:
        try:
            img_response = await requests_session.request(
                method = 'GET', 
                url = url_img, 
                headers = {'User-Agent': USER_AGENT}, 
                allow_redirects = True, 
                ssl = False)
            assert img_response.status == 200
            img_bytes = await img_response.read()
        except:
            img_bytes = b''
        try:
            img = Image.open(BytesIO(img_bytes))
            if np.prod(img.size) > Image.MAX_IMAGE_PIXELS:
                continue
            img = img.convert('RGBA')
            text_in_img = pytesseract.image_to_string(img, lang = 'eng')
        except Exception as e:
            text_in_img = ''
        text_in_imgs.append(text_in_img)
    
    # Remove empty strings
    text_in_imgs = list(filter(None, text_in_imgs))
    
    return text_in_imgs

async def get_html_content(url: str, file_path: str, browser_context: BrowserContext) -> str:
    """
    Visit the URL and save a PDF and HTML version of the website locally
    """

    # TODO
    # Check whether the request has been redirected and report also the actual URL
    # The EXCLUDED rule should look at both the URLs

    html_path = file_path.replace('.pdf', '.html')

    if not os.path.exists(html_path):
        page = await browser_context.new_page()
        
        try:
            await page.goto(url, wait_until='networkidle')
        except:
            return None
        
        if HEADLESS and not os.path.exists(file_path):
            try:
                # TODO 
                # There are cases in which the text is in gray and is unreadable by the OCR (assuming it has to be used on the file)
                # See this example: https://www.surfacetechnology.com/Products-and-Services/Products/Addplate-and-NiPLATE-730-Chemical-Solutions.aspx
                await page.pdf( 
                    path = file_path, 
                    margin = {side: PDF_MARGIN \
                        for side in ['top', 'right', 'bottom', 'left']})
            except:
                pass
        
        try:
            html = await page.content()
            await page.close()
        except:
            return None
        else:
            with open(html_path, 'w') as f_out:
                f_out.write(html)
    
    return html_path

async def get_text_from_html(url: str, file_path: str, requests_session: ClientSession, browser_context: BrowserContext, use_ocr: bool = False) -> Tuple[str, List[str]]:
    """ 
    Extract the text from the body of the document, 
      using the local version of the website (or try to create one)
    """

    html_path = await get_html_content(
        url = url, 
        file_path = file_path, 
        browser_context = browser_context)
    
    # Use the, previously stored, local HTML version of the URL, if exists
    if html_path and os.path.exists(html_path):
        if use_ocr:
            if os.path.exists(file_path):
                text = await get_text_from_pdf(
                    url = url, 
                    file_path = file_path, 
                    requests_session = requests_session, 
                    use_ocr = True)
                text_in_imgs = []
                return text, text_in_imgs
            else:
                return '', []
        
        with open(html_path, 'r') as f_in:
            try:
                html_soup = beautiful_soup(f_in, 'html5lib')
            except:
                text = ''
                url_imgs = []
            else:
                try:
                    # Remove <script> and <style> tags
                    script_style = [el.extract() \
                        for tag in ['script', 'style'] \
                            for el in html_soup.find_all(tag)]
                    # Extract text from the <body> of the page
                    body = html_soup.find('body')
                    # Remove hidden elements
                    hidden_elements = [el.extract() \
                        for el in body.find_all(
                            style=re.compile(f'display:\s*none'))]
                    # TODO There are cases like
                    #      https://www.flir.com/patentnotices/instruments/
                    #      https://www.eppendorf.com/fileadmin/General/trademarks-patents/patents_us.htm
                    #      in which part of the relevant content is contained 
                    #      in an iFrame tag. It would be desirable to extract
                    #      it and store it in the local static version of
                    #      the document
                    text = body.get_text(separator = ' ')
                except:
                    text = ''
                try:
                    url_imgs = [img.get('src') \
                        for img in html_soup.find_all('img') if img.get('src')]
                except:
                    url_imgs = []
        text_in_imgs = await get_text_from_imgs(
            url = url, 
            url_imgs = url_imgs, 
            requests_session = requests_session)

        return text, text_in_imgs
    return '', []

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
            file_path = file_path, 
            requests_session = requests_session)
    
    try:
        text = rtf_to_text(text_bytes.decode(errors='ignore'))
    except:
        text = ''

    return text

async def get_text(url: str, file_path: str, requests_session: ClientSession, browser_context: BrowserContext, use_ocr: bool = None) -> Tuple[List[str], List[str], str]:
    """ 
    Extract text from the URL provided
    Return both the document content and the document type
    Note: When relevant, use also the OCR
    """

    content_type = await get_content_type(
        url = url, 
        file_path = file_path, 
        requests_session = requests_session)
    
    if content_type == 'PDF':
        text = await get_text_from_pdf(
            url = url, 
            file_path = file_path, 
            requests_session = requests_session, 
            use_ocr = False)
        text_ocr = await get_text_from_pdf(
            url = url, 
            file_path = file_path, 
            requests_session = requests_session, 
            use_ocr = True)
        texts = [text, text_ocr]
        text_in_imgs = []
        cookie_found = False
    elif content_type == 'TXT':
        file_path = file_path.replace('.pdf', '.txt')
        text = await get_text_from_txt(
            url = url, 
            file_path = file_path, 
            requests_session = requests_session)
        texts = [text]
        text_in_imgs = []
        cookie_found = False
    elif content_type == 'RTF':
        file_path = file_path.replace('.pdf', '.rtf')
        text = await get_text_from_rtf(
            url = url,
            file_path = file_path,
            requests_session = requests_session)
        texts = [text]
        text_in_imgs = []
        cookie_found = False
    elif content_type == 'HTML':
        text, text_in_imgs = await get_text_from_html(
            url = url, 
            file_path = file_path, 
            requests_session = requests_session, 
            browser_context = browser_context, 
            use_ocr = use_ocr)
        # Remove sentences if 'cookie' or 'privacy policy' are named
        cookie_found = COOKIE_RE.search(text) is not None
        text = re.sub(' \.+', '', 
            ' '.join([sentence \
                for sentence in sent_tokenize(re.sub('\n{2,}', '. ', text)) \
                    if not (COOKIE_RE.search(sentence) and len(sentence)<CONTEXT_SPAN)]))
            # FIXME The '\n{2,}' regex is problematic in this case
            #       https://www.optoknowledge.com/mstir.html
            #       It would be better to use '(\n\s?){2,}'
            #       However, it would be better to test it on a larger 
            #       sample to see if there are unforeseen disadvantages 
            #       with this new regex
        texts = [text]
    elif content_type == 'OTHER':
        file_path = file_path.replace('.pdf', '.other')
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f_out:
                f_out.write('')
        texts = []
        text_in_imgs = []
        cookie_found = False
    else:
        texts = []
        text_in_imgs = []
        cookie_found = False

    # Remove new lines, tabs and spaces
    texts = [PUNCT_RE.sub(' ', text).strip() for text in texts]
    text_in_imgs = [PUNCT_RE.sub(' ', text_in_img).strip() \
        for text_in_img in text_in_imgs]
    # Remove empty strings
    text_in_imgs = list(filter(None, text_in_imgs))

    return texts, text_in_imgs, content_type, cookie_found

def merge_matches(matches: List[Tuple[int]]) -> Generator[Tuple[int], None, None]:
    """ 
    If two or more of the intervals provided overlap, join them in one
    """

    matches.sort()
    
    start, end = matches[0]
    for start_next, end_next in matches[1:]:
        if (start_next-1) <= end:
            end = max(end, end_next)
        else:
            yield start, end
            start, end = start_next, end_next
    yield start, end

async def get_corpus(url: str, patent_ids_regex: Pattern, file_path: str, requests_session: ClientSession, browser_context: BrowserContext, use_ocr: bool = None) -> Tuple[List[str], List[str], List[str], List[str], str]:
    """ 
    Using several regular expressions, extract the relevant corpus 
      (i.e., 500 characters around the keyword considered) 
      from the text extracted from the document
    """
    
    # TODO Consider using fuzzy matching with the NUMPAT_RE to extract also cases with one (or few) errors
    #      The 'regex' package replaces 're' and has also fuzzy matching

    PATNUM_RE = patent_ids_regex
    texts, text_in_imgs, content_type, cookie_found = await get_text(
        url = url, 
        file_path = file_path, 
        requests_session = requests_session, 
        browser_context = browser_context, 
        use_ocr = use_ocr)
    
    headers = set()
    corpus = set()
    footers = set()
    for text in texts:
        corpus_matches = [match.span() \
            for CORPUS_RE in CORPUS_RES+[PATNUM_RE] \
                for match in CORPUS_RE.finditer(text)]
        
        corpus_matches_len = len(corpus_matches)

        if corpus_matches_len == 0 and \
           not any(PATNUM_RE.search(text_in_img) \
               for text_in_img in text_in_imgs) and \
           content_type == 'HTML' and \
           not use_ocr:
            return await get_corpus(
                url = url, 
                patent_ids_regex = patent_ids_regex, 
                file_path = file_path, 
                requests_session = requests_session, 
                browser_context = browser_context, 
                use_ocr = True)
        
        if corpus_matches_len>1:
            # NB merge_matches returns a generator, not a list
            corpus_matches = list(merge_matches(corpus_matches))

        for start, end in corpus_matches:
            start = round(max(0,start-(CONTEXT_SPAN/2)))
            end = round(end+(CONTEXT_SPAN/2))
            matched = text[start:end].strip()
            corpus.add(matched)
        
        header = text[:round(CONTEXT_SPAN/4)].strip()
        headers.add(header)

        footer = text[-round(CONTEXT_SPAN/4):].strip()
        footers.add(footer)

    headers = list(filter(None, headers))
    corpus = list(filter(None, corpus))
    footers = list(filter(None, footers))
    
    return headers, corpus, footers, text_in_imgs, content_type, cookie_found

async def test_rule(rule_label: str, doc_info: DocInfoDict) -> bool:
    """ 
    Apply a rule using the relevant information from the document provided
    """

    try:
        rule = RULE_FS[rule_label]
        is_rule = await rule(doc_info)
    except:
        is_rule = 'EXCEPTION'
    finally:
        return is_rule

async def parse_urls_and_write_results(line_idx: int, urls: List[str], patent_ids: List[int], exclude_websites: List[str], files_folder: str, requests_session: ClientSession, playwright: ContextManager, out_path: PosixPath) -> bool:
    """ 
    Get the document from the URL, extract the relevant texts from the document, 
      apply the classification rules and write the results in the output file
    """

    patent_ids_regex = generate_patent_ids_regex(patent_ids)

    browser = await playwright.chromium.launch(args = BROWSER_CONFIG, headless = HEADLESS)
        
    browser_context = await browser.new_context(**CONTEXT)
    browser_context.set_default_timeout(TIMEOUT)

    urls_len = len(urls)
    for idx, url in enumerate(tqdm(urls, position = 1, desc = f'Line {line_idx}', leave = False)):
        url_out = dict()
        url_out[URL_LABEL] = url
        url_out[PATENT_ID_LABEL] = patent_ids
        
        url = re.sub(r'/$', '', url)

        file_name = generate_file_name(url, files_folder)
        file_path = f'{files_folder}/{file_name}.pdf'

        headers, corpus, footers, text_in_imgs, content_type, cookie_found = await get_corpus(
            url, patent_ids_regex, file_path, requests_session, browser_context)

        url_out['CONTENT_TYPE'] = content_type
        url_out['HEADERS'] = [''] if len(headers)==0 else headers
        url_out['CORPUS'] = [''] if len(corpus)==0 else corpus
        url_out['FOOTERS'] = [''] if len(footers)==0 else footers
        url_out['TEXT_IN_IMGS'] = [''] if len(text_in_imgs)==0 else text_in_imgs
        url_out['COOKIE_FOUND'] = cookie_found
        
        doc_info = {
            'url': url,
            'content_type': content_type,
            'headers': headers,
            'corpus': corpus,
            'footers': footers,
            'text_in_imgs': text_in_imgs,
            'patent_ids_regex': patent_ids_regex,
            'exclude_websites': exclude_websites}
        rules = dict()
        for rule_label in RULE_FS.keys():
            rule_outcome = await test_rule(rule_label, doc_info)
            rules[rule_label] = rule_outcome

        url_out.update(rules)
        url_out = json.dumps(url_out)

        async with aiofiles.open(out_path, 'a') as f_out:
            await f_out.write(url_out+'\n')
        
        # If there are more than 5 URLs in one of the lines and it is not the last URL of the line
        extra_wait = urls_len >= 5 and idx+1 != urls_len
        if extra_wait:
            # Wait 10' more, so that the script doesn't overload the requested website
            await sleep(10)
    
    await browser_context.close()
    await browser.close()

    return True


#################
#  PARALLELIZE  #
#################

async def run_task(line_idx: int, line: LineDict, exclude_websites: List[str], files_folder: str, requests_session: ClientSession, playwright: ContextManager, out_path: PosixPath) -> None:
    """ 
    Run the parser and writer asynchronously 
      (limiting to 50 the maximum number of tasks run at a time)
    """

    async with SEMAPHORE:
        results = await parse_urls_and_write_results(
            line_idx = line_idx,
            urls = line['urls'], 
            patent_ids = line['patent_ids'], 
            exclude_websites = exclude_websites, 
            files_folder = files_folder, 
            requests_session = requests_session, 
            playwright = playwright,
            out_path = out_path)
        if not results:
            return None


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

    with open(pwd.joinpath(args.input_list[0]), 'r') as f_in:
        data = [json.loads(line) for line in f_in.read().splitlines()]
    
    # Take only the relevant information from the database
    data = [{key: value \
        for key, value in line.items() \
            if key in ['vpm_pages', 'patent_id']} \
                for line in data]

    # Remove the lines with no VPM page from the database
    data = [line for line in data if len(list(filter(None, line['vpm_pages'])))>0]

    # Group lines that share a page (or more) together
    data = deduplicate_data(data)
    
    # Read the list of websites that have been excluded
    with open(pwd.joinpath(args.input_list[1]), 'r') as f_in:
        exclude_websites = f_in.read().splitlines()
    
    # If you set DEBUGGING as True in the settings, use only a random sample of the data
    if DEBUGGING:
        import random
        data = random.sample(data, SAMPLE_SIZE)
        with open('sample.jsonl', 'w') as f_sample:
            f_sample.write('\n'.join([json.dumps(line) for line in data]))

    out_path = pwd.joinpath(args.output)
    
    # Remove the VPM pages already classified
    # Note: the new data are appended to the output file, provided it exists 
    #   (please, don't use a file you want to preserve or overwrite)
    already_classified_urls = []
    if os.path.exists(out_path):
        with open(out_path, 'r') as f_bak:
            already_classified_urls = [json.loads(line)[URL_LABEL] \
                for line in f_bak.read().splitlines()]
    data_left = []
    for line in data:
        patent_ids = line['patent_id']
        vpm_pages = line['vpm_pages']
        vpm_pages = [vpm_page for vpm_page in vpm_pages \
            if vpm_page not in already_classified_urls]
        if len(vpm_pages)>0:
            line_left = {'urls': vpm_pages, 'patent_ids': patent_ids}
            data_left.append(line_left)

    # Create the folder in which the PDF, HTML and TXT files downloaded (or generated) by the script are stored
    # These files will also be used (preferably) by the script if they exist 
    #   (i.e., in that case, the document will not be checked online again)
    files_folder = 'files'
    if not os.path.exists(files_folder):
        os.mkdir(files_folder)

    async with ClientSession() as requests_session, \
               async_playwright() as playwright:
        
        # Create a task for each line
        tasks = [asyncio.ensure_future(
            run_task(
                line_idx = line_idx,
                line = line, 
                exclude_websites = exclude_websites, 
                files_folder = files_folder, 
                requests_session = requests_session, 
                playwright = playwright,
                out_path = out_path)) \
                    for line_idx, line in enumerate(data_left)]
        
        # Run the tasks
        for task in tqdm(asyncio.as_completed(tasks), total = len(tasks), desc = 'Main'):
            await task


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
