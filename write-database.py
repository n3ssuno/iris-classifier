#!/usr/bin/env python

"""
Script to populate a database useful for the classifier
It merges and arranges information from the input and output 
 of the pre-classifier in a convenient format for the classifier

Author: Carlo Bottai
Copyright (c) 2021 - TU/e and EPFL
License: See the LICENSE file.
Date: 2021-04-10

"""


## LIBRARIES ##

import numpy as np
import pandas as pd
import json
from flata import Flata, JSONStorage
from iris_utils.parse_args import parse_io


def main():
    args = parse_io()

    # Read the input of the pre-classifier
    data = pd.read_json(
        args.input_list[0], 
        lines=True) \
        .explode('vpm_pages') \
        .rename(columns={
            'vpm_pages': 'vpm_page'})
    data['vpm_page'] = data.vpm_page \
        .apply(lambda row: np.nan if row is None else row)
    data['scraped_websites'] = data.scraped_websites \
        .apply(lambda row: [np.nan] if row[0] is None else row)
    
    # Read the output of the pre-classifier
    automatic_classification = pd.read_json(
        args.input_list[1], 
        lines=True) \
        .rename(columns={
            'VPM_PAGE': 'vpm_page'})

    automatic_classification_list = []

    # Classify as false cases those pages that have been identified as 
    #   patents, SEC documents, or as part of irrelevant domains 
    query = ' | '.join([f'{rule}==True' \
        for rule in ['EXCLUDED', 'PATENT', 'SEC']])
    false_vpm_pages = automatic_classification.query(query)[['vpm_page']]
    if len(false_vpm_pages):
        false_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | False patent-product link'
        automatic_classification_list.append(false_vpm_pages)
    
    # Classify as true cases those pages for which 
    #   (at least) one of the "strong" rules is True
    query_pos = ' | '.join([f'{rule}==True' \
        for rule in ['URL', 'LAW', 'TRADEMARK', 'TEXT']])
    query_neg = ' & '.join([f'{rule}==False' \
        for rule in ['EXCLUDED', 'PATENT', 'SEC']])
    query = f'({query_pos}) & {query_neg}'
    true_vpm_pages = automatic_classification.query(query)[['vpm_page']]
    if len(true_vpm_pages):
        true_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | True patent-product link'
        automatic_classification_list.append(true_vpm_pages)
    
    # Classify as COPYRIGHT those pages for which 
    #   the COPYRIGHT rule is True 
    # This label will be used by the classifier
    query = ' & '.join([f'{rule}==False' \
        for rule in [
            'EXCLUDED', 
            'PATENT', 
            'SEC', 
            'URL', 
            'LAW', 
            'TRADEMARK', 
            'TEXT']])
    copyright_vpm_pages = automatic_classification \
        .query(f'COPYRIGHT==True & {query}')[['vpm_page']]
    if len(copyright_vpm_pages):
        copyright_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | COPYRIGHT'
        automatic_classification_list.append(copyright_vpm_pages)
    
    # Classify as NOCORPUS+IMG those pages for which 
    #   both the NOCORPUS and IMG rules are True
    # This label will be used by the classifier
    query = ' & '.join([f'{rule}==False' \
        for rule in [
            'EXCLUDED', 
            'PATENT', 
            'SEC', 
            'URL', 
            'LAW', 
            'TRADEMARK', 
            'TEXT', 
            'COPYRIGHT']])
    nocorpus_imgs_vpm_pages = automatic_classification \
        .query(f'NOCORPUS==True & IMG==True & {query}')[['vpm_page']]
    if len(nocorpus_imgs_vpm_pages):
        nocorpus_imgs_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | NOCORPUS+IMG'
        automatic_classification_list.append(nocorpus_imgs_vpm_pages)
    
    # Classify as NOCORPUS+PATNUMINURL those pages for which 
    #   both the NOCORPUS and PATNUMINURL rules are True
    # This label will be used by the classifier
    query = ' & '.join([f'{rule}==False' \
        for rule in [
            'EXCLUDED', 
            'PATENT', 
            'SEC', 
            'URL', 
            'LAW', 
            'TRADEMARK', 
            'TEXT', 
            'COPYRIGHT',
            'IMG']])
    nocorpus_patnuminurl_vpm_pages = automatic_classification \
        .query(f'NOCORPUS==True & PATNUMINURL==True & {query}')[['vpm_page']]
    if len(nocorpus_patnuminurl_vpm_pages):
        nocorpus_patnuminurl_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | NOCORPUS+PATNUMINURL'
        automatic_classification_list.append(nocorpus_patnuminurl_vpm_pages)
    
    # Classify as NOCORPUS those pages for which 
    #   the NOCORPUS rule is True and the IMG rule is False
    # This label will be used by the classifier
    query = ' & '.join([f'{rule}==False' \
        for rule in [
            'EXCLUDED', 
            'PATENT', 
            'SEC', 
            'URL', 
            'LAW', 
            'TRADEMARK', 
            'TEXT', 
            'COPYRIGHT',
            'IMG',
            'PATNUMINURL']])
    nocorpus_vpm_pages = automatic_classification \
        .query(f'NOCORPUS==True & {query}')[['vpm_page']]
    if len(nocorpus_vpm_pages):
        nocorpus_vpm_pages.loc[:,'vpm_page_automatic_classification'] = \
            'Automatic classification | NOCORPUS'
        automatic_classification_list.append(nocorpus_vpm_pages)
    
    # Put together the labels just created
    automatic_classification = pd.concat(automatic_classification_list)
    
    # Create another column with the definitive classification
    # For the pages that have been labeled as surely true (or false), report this same label
    # For the unsure pages, report None
    automatic_classification['vpm_page_classification'] = np.nan
    subset = automatic_classification.vpm_page_automatic_classification \
        .str.split(' \| ').str[1].isin([
            'True patent-product link', 
            'False patent-product link'])
    automatic_classification.loc[subset, 'vpm_page_classification'] = \
        automatic_classification.loc[subset, 'vpm_page_automatic_classification']
    
    # Merge the labels just created 
    #   with the other information pieces from the main database
    data_out = pd.merge(
        data, automatic_classification, 
        on='vpm_page', how='left')

    subset = data_out.vpm_page.isna()
    
    # Label as Unclassified the rows still without a classification
    data_out.loc[
        ~subset, 'vpm_page_automatic_classification'] = \
            data_out \
                .loc[~subset, 'vpm_page_automatic_classification'] \
                .fillna('Automatic classification | Unclassified')

    # Reshuffle randomly the data
    data_out = data_out.sample(frac=1, random_state=410)
    
    out_name = args.output.split('.')
    out_base = '.'.join(out_name[:-1])
    out_ext = out_name[-1]
    
    frac_size = 1/args.n_output
    
    for idx in range(args.n_output):
        if idx<args.n_output-1:
            data_out_frac = data_out.sample(frac=frac_size)
            data_out = data_out.drop(data_out_frac.index)
        else:
            data_out_frac = data_out
        
        # Transform the DataFrame into a list of dictionaries
        data_out_frac = json.loads(data_out_frac.to_json(orient='records'))
        
        # Create the output database
        DB = Flata(f'{out_base}_{idx}.{out_ext}', storage=JSONStorage)

        # Create an output table into the database
        database = DB.table('iris_vpm_pages_classifier')

        # Populate the database with the useful data
        added_data = database.insert_multiple(data_out_frac)


if __name__ == '__main__':
    main()
