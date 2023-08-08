#!/usr/bin/env python
# coding: utf-8

import subprocess
import pkgutil

packages_to_install = [
    'numpy', 'python-dotenv', 'pandas', 'matplotlib', 'requests', 'bs4', 'lxml'
]

for package in packages_to_install:
    if not pkgutil.find_loader(package):
        subprocess.run(['pip', 'install', package, '--quiet'])


import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
import sys



base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
api_key = 'cf94d59a96d4b17b7389dde3a72724a2eb08'



def search_by_terms(terms, db='pubmed', retmax=1000, use_history=True):

    if not terms:
        print('ERROR: No terms to search for')
        if use_history:
            return None, None, None
        return None
    
    terms_string = '+AND+'.join([s.strip().replace(' ', '+') for s in terms])

    url = f'{base_url}esearch.fcgi?db={db}&term={terms_string}&retmax={retmax}&api_key={api_key}'
    if use_history:
        url += '&usehistory=y'

    response = requests.get(url)
    if response.status_code != 200:
        print('ERROR: Bad response code')
        if use_history:
            return None, None, None
        return None
    
    ids = re.findall(r"<Id>(\d+)</Id>", response.text)

    if use_history:
        web_match = re.search(r"<WebEnv>(\S+)</WebEnv>", response.text)
        web = web_match.group(1) if web_match else None

        key_match = re.search(r"<QueryKey>(\d+)</QueryKey>", response.text)
        key = key_match.group(1) if key_match else None

        return ids, web, key

    return ids



def get_articles_data(ids=[], web_env='', query_key='', db='pubmed', retmax=1000):

    use_web_env = False

    if not ids and not (query_key and web_env):
        print('ERROR: No ids or query_key/web_env provided')
        return None
    elif not ids:
        use_web_env = True 

    url = f'{base_url}efetch.fcgi?db={db}'
    if use_web_env:
        url += f'&query_key={query_key}&WebEnv={web_env}'
    else:
        ids_string = [str(id) for id in ids]
        url += '&id=' + ','.join(ids_string)

    url += f'&rettype=abstract&retmode=xml&api_key={api_key}&retmax={retmax}'

    response = requests.get(url)

    if response.status_code != 200:
        print('ERROR: Bad response code')
        return None
    
    soup = BeautifulSoup(response.text, features="xml")
    articles = soup.find_all('PubmedArticle')
    if not articles:
        print('ERROR: No articles found')
        return None
    
    data = pd.DataFrame(columns=['id', 'title', 'abstract', 'keywords', 'pub_date'])
    for article in articles:
        id = article.find('PMID').get_text()
        date = article.find('PubMedPubDate', {'PubStatus': 'received'})
        pub_date = None
        if date:
            pub_date = datetime.strptime(f'{date.find("Day").get_text()} {date.find("Month").get_text()} {date.find("Year").get_text()}', "%d %m %Y")
        title = article.find('ArticleTitle').get_text()
        abstract = ''.join([a.get_text() for a in article.find_all('AbstractText')])
        keywords = [k.get_text() for k in article.find_all('Keyword')]
        data = pd.concat([data, pd.DataFrame({'id': id, 'title': title, 'abstract': abstract, 'keywords': [keywords], 'pub_date': pub_date})]).reset_index(drop=True)

    return data


def clean_data(data, drop_id_duplicates=True, drop_empty_abstracts=True, drop_nan_abstracts=True, drop_abstracts_with_matches=True, drop_abstracts_matches=['[This corrects the article DOI: ', '[This retracts the article DOI: '], drop_date_nan=False, drop_date_before=None, drop_date_after=None, search_terms=[]):
    if data is None or data.empty:
        print('ERROR: No data provided')
        return None

    if drop_id_duplicates:
        data = data.drop_duplicates(subset=['id']).reset_index(drop=True)

    if drop_empty_abstracts:
        data = data.loc[data['abstract'] != ''].reset_index(drop=True)

    if drop_nan_abstracts:
        data = data.dropna(subset=['abstract']).reset_index(drop=True)
    
    if drop_abstracts_with_matches and drop_abstracts_matches:
        data = data.loc[~data['abstract'].str.startswith(tuple(drop_abstracts_matches))].reset_index(drop=True)  

    if drop_date_nan:
        data = data.dropna(subset=['pub_date']).reset_index(drop=True)

    if drop_date_before:
        data = data.loc[data['pub_date'] > drop_date_before].reset_index(drop=True)
    if drop_date_after:
        data = data.loc[data['pub_date'] < drop_date_after].reset_index(drop=True)
    
    if search_terms:
        data['search_terms'] = [search_terms]*len(data)
        
    return data


def data_extraction_pipeline(terms, db='pubmed', n_articles=1000, use_history=True, drop_id_duplicates=True, drop_empty_abstracts=True, drop_nan_abstracts=True, drop_abstracts_with_matches=True, drop_abstracts_matches=['[This corrects the article DOI: ', '[This retracts the article DOI: '], drop_date_nan=False, drop_date_before=None, drop_date_after=None, add_search_terms=True, file_name=None):
    data = None
    if use_history:
        ids, web, key = search_by_terms(terms, db=db, retmax=n_articles, use_history=use_history)
        data = get_articles_data(web_env=web, query_key=key, retmax=n_articles)
    else:
        ids = search_by_terms(terms, db=db, retmax=n_articles, use_history=use_history)
        data = get_articles_data(ids=ids, db=db, retmax=n_articles)

    if data is None:
        return None
    
    if add_search_terms:
        data = clean_data(data, drop_id_duplicates=drop_id_duplicates, drop_empty_abstracts=drop_empty_abstracts, drop_nan_abstracts=drop_nan_abstracts, drop_abstracts_with_matches=drop_abstracts_with_matches, drop_abstracts_matches=drop_abstracts_matches, drop_date_nan=drop_date_nan, drop_date_before=drop_date_before, drop_date_after=drop_date_after, search_terms=terms)
    else:
        data = clean_data(data, drop_id_duplicates=drop_id_duplicates, drop_empty_abstracts=drop_empty_abstracts, drop_nan_abstracts=drop_nan_abstracts, drop_abstracts_with_matches=drop_abstracts_with_matches, drop_abstracts_matches=drop_abstracts_matches, drop_date_nan=drop_date_nan, drop_date_before=drop_date_before, drop_date_after=drop_date_after)
    
    if file_name:
        data.to_csv(f'../data/{file_name}.csv', index=False)
        
    return data

def main(return_data=True):
    terms = []
    print('Enter search terms (enter "q" when finished):')
    while True:
        term = input()
        if term == 'q':
            break
        if term != '':
            terms.append(term)

    if not terms:
        print('ERROR: No terms provided')
        sys.exit()

    print('Enter max number of articles to search for:')
    n_articles = 0
    while (n_articles < 1):
        try:
            n_articles = int(input())
        except:
            print('ERROR: Invalid input')

    print(f'Searching for max {n_articles} articles with terms: {terms}')

    print('Do you want to save the data? (y/n)')
    while True:
        save = input()
        if save == 'y':
            # print('Enter file name:')
            # file_name = input()
            # file_name = file_name.replace(' ', '_')
            file_name = 'data'
            data = data_extraction_pipeline(terms=terms, n_articles=n_articles, file_name=file_name)
            break
        elif save == 'n':
            data = data_extraction_pipeline(terms=terms, n_articles=n_articles)
            break
    if return_data:
        return data
