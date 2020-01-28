import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.special import loggamma
from scipy.stats import beta
from scipy.optimize import fsolve
from sklearn.utils import resample
import imdb
from os import path

movie_list = pd.read_csv('movie_list.csv')
code_list = movie_list['Code']
movie_dict = {}
if path.exists('movie_dict.pickle'):
    with open('movie_dict.pickle', 'rb') as f:
        movie_dict = pickle.load(f) # movie_dict is a dict with key: movie code, value: rating and credible interval

for index, code in enumerate(code_list):
    if code not in movie_dict:
        data = pd.read_csv('{}.csv'.format(code), index_col=0) # reads in helpfulness data for that movie
        data = imdb.transform_data(data) # transform_data computes the beta prior for each review and finds the posterior
        my_rating = imdb.compute_rating(data) # computes rating that is weighted based on estimated helpfulness
        credible_interval = imdb.credible_interval(data) # finds a 95% credible interval for the rating
        movie_dict[code] = [my_rating, credible_interval]
        with open('movie_dict.pickle', 'wb') as f:
            pickle.dump(movie_dict, f)
    print(index + 1)

# Additional data wrangling
ratings = pd.DataFrame(movie_dict).T
movie_list = movie_list.merge(ratings, left_on='Code', right_on=ratings.index)
movie_list = movie_list.rename({'Unnamed: 0': 'Old Ranking', 0:'My Rating', 1:'Confidence Interval'}, axis=1)
movie_list['Lower Bound'] = movie_list.apply(lambda x: x['Confidence Interval'][0], axis=1)
movie_list['Upper Bound'] = movie_list.apply(lambda x: x['Confidence Interval'][1], axis=1)
movie_list.drop('Confidence Interval', inplace=True, axis=1)
movie_list=movie_list.sort_values('Lower Bound', ascending=False)
movie_list['New Ranking']=np.arange(len(movie_list))
movie_list['Change Ranking']=movie_list['New Ranking'] - movie_list['Old Ranking']
movie_list.to_csv("final_ratings.csv")
