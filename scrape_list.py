import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

# This file scrapes a list of movies from the IMDB top 1000 movies list

code_list = []
movie_list = []
year_list = []
rating_list = []
for start in range(1, 1001, 50):
    url = 'https://www.imdb.com/search/title/?groups=top_1000&view=simple&sort=user_rating,desc&start={}&ref_=adv_nxt'.format(start)
    response = requests.get(url)
    parser = BeautifulSoup(response.content.decode('utf-8'), 'html.parser')
    list_parse = parser.find_all('h3', class_='lister-item-header')


    for item in list_parse:
        code = item.find_all('a')[0]['href'][-10:-1]
        movie = item.find_all('a')[0].text
        year = item.find_all('span', class_='lister-item-year text-muted unbold')[0].text[-5:-1]
        code_list.append(code)
        movie_list.append(movie)
        year_list.append(year)

    list_parse = parser.find_all('div', class_='inline-block ratings-imdb-rating')
    for item in list_parse:
        rating = item['data-value']
        rating_list.append(rating)

movie_df = pd.DataFrame([code_list, movie_list, year_list, rating_list]).T
movie_df.columns = ['Code', 'Name', 'Year', 'IMDB_Rating']
movie_df.to_csv('movie_list.csv')
