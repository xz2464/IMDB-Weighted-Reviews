import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from scipy.special import loggamma
from scipy.stats import beta
from scipy.optimize import fsolve
from sklearn.utils import resample
import statsmodels.api as sm

def scrape_helpfulness(movie_code):
    """
    Scrapes the review pages for the movie and, for each review, records the
    number of stars, the number of people who found it helpful, and the total
    number of votes for that review.

    Parameters:
    movie_code (str): Unique IMDB code for movie/tv

    Returns:
    df: DataFrame where each row is a review, records number of stars and helpfulness
    """
    reviews = []
    data_key = ''
    if path.exists('{}.csv'.format(movie_code)): # check if file exists, if so, then return dataframe.
        return pd.read_csv('{}.csv'.format(movie_code))
    while True:
        response = requests.get('https://www.imdb.com/title/{}/reviews/_ajax?ref_=undefined&paginationKey={}'.format(movie_code, data_key))
        parser = BeautifulSoup(response.content, 'html.parser')
        review_parse = parser.find_all('div', class_='lister-item-content')
        for index, review in enumerate(review_parse):
            user_rating = review.find_all('span', class_='rating-other-user-rating')
            if user_rating: # records the star and helpfulness of the review
                star = int(user_rating[0].find_all('span')[0].text)
                helpfulness = review.find_all('div', class_='actions text-muted')[0].text
                reviews.append(pd.Series([star, helpfulness]))
        data_key = parser.find_all('div', class_='load-more-data')
        if not data_key: # this indicates there are no more reviews left
            break
        data_key = data_key[0]['data-key'] # this is a key used to get to the next page for scraping
    reviews_df=pd.concat(reviews, axis=1).T
    reviews_df.columns = ['stars', 'helpfulness']
    cleaned_reviews_df = reviews_df.copy()

    # extracts the helpfulness votes
    pattern = r'\b(\d,?\d*)\b'
    helpfulness_ratings = cleaned_reviews_df['helpfulness'].str.extractall(pattern).unstack()
    helpfulness_ratings.columns=helpfulness_ratings.columns.get_level_values(1)
    helpfulness_ratings.rename({0: 'helpful', 1:'total votes'}, axis=1, inplace=True)

    # data wrangling and save csv file
    cleaned_reviews_df=cleaned_reviews_df.merge(helpfulness_ratings, left_index=True, right_index=True)
    cleaned_reviews_df['helpful'] = cleaned_reviews_df['helpful'].str.replace(',', '')
    cleaned_reviews_df['total votes'] = cleaned_reviews_df['total votes'].str.replace(',', '')
    cleaned_reviews_df['stars'] = pd.to_numeric(cleaned_reviews_df['stars'])
    cleaned_reviews_df['helpful'] = pd.to_numeric(cleaned_reviews_df['helpful'])
    cleaned_reviews_df['total votes'] = pd.to_numeric(cleaned_reviews_df['total votes'])
    cleaned_reviews_df['helpfulness'] = cleaned_reviews_df['helpful']/cleaned_reviews_df['total votes']
    cleaned_reviews_df.to_csv('{}.csv'.format(movie_code))
    print('{} done'.format(movie_code))
    return cleaned_reviews_df

def scrape_all_code(code_list):
    """
    Scrapes all movie reviews from every movie in the code list and computes helpfulness

    Parameters:
    code_list (list): list of IMDB codes

    Returns:
    None
    """
    counter = 0
    for code in code_list:
        scrape_helpfulness(code)
        counter += 1
        print(counter)

def compute_parameters(p, mean, variance):
    """
    Given mean and variance of beta distribution, compute its parameters

    Parameters:
    p (float, float): starting point of optimization
    mean (float): Mean of beta
    variance (float): Variance of beta

    Returns:
    (float, float): Used for optimization
    """
    a, b = p
    return (a/(a+b)-mean, (a*b)/((a+b)**2*(a+b+1))-variance)

def update_params(a, b, helpful_votes, total_votes):
    """
    Updates parameters for beta posterior

    Parameters:
    a (float): prior parameter
    b (float): prior parameter
    helpful_votes (int): num of helpful notes
    total_votes (int): total number of votes

    Returns:
    (list): posterior parameters
    """
    a_new = a + helpful_votes
    b_new = b + total_votes - helpful_votes
    return [a_new, b_new]

def estimated_helpfulness(a, b):
    """Uses mean of beta posterior to estimate helpfulness"""
    return beta(a, b).mean()

def find_parameters(params, star):
    """Finds the beta paremeters for the prior given the number of stars"""
    if star in params.index:
        return params.loc[star]
    else:
        return [1, 1] # If doesn't exist, assume beta parameters are 1, 1

def transform_data(df):
    """Computes beta prior and posterior and computes an estimated helpfulness score
    Parameters:
    df (DataFrame): DataFrame of reviews consisting of helpfulness and total votes

    Returns:
    (DataFrame): Consisting of new Beta parameters and Estimated Helpfulness Rankings
    """

    # We first fit a binomial model to estimate the helpfulness based on number of stars
    non_null = df[df['helpfulness'].notnull()].copy()
    binom_model = sm.GLM(non_null['helpfulness'], non_null['stars'], family=sm.families.Binomial())
    results = binom_model.fit()
    means = results.predict(np.arange(1,11)) # predicted mean score based on number of stars

    # solves for parameters based on mean (derived from binomial model). Variance made as large as possible
    params = pd.Series(means).map(lambda x: fsolve(compute_parameters, (0.5,0.5), args=(x,x*(1-x) - 0.03)))

    # finds beta parameters, updates to get posterior, then computes helpfulness
    df['old_params'] = df.apply(lambda x: find_parameters(params, x[0]), axis=1)
    df['new_params'] = df.apply(lambda row: update_params(row[4][0], row[4][1], row[2], row[3]), axis=1)
    df['estimated_helpfulness'] = df.apply(lambda row: estimated_helpfulness(row[5][0], row[5][1]), axis=1)
    return df

def compute_rating(df):
    """ Computes a weighted rating based on the estimated helpfulness
    Parameters:
    (DataFrame): DF consisting of estimated helpfulness scores

    Returns:
    (float): Weighted score
    """
    weight = df['estimated_helpfulness']/df['estimated_helpfulness'].sum()
    weighted_stars = weight*df['stars']
    return round(weighted_stars.sum(), 1)

def credible_interval(df):
    """ Gives a `confidence interval` based on bootstrapping from data and sampling from beta posteriors
    Parameters:
    (df): DataFrame of reviews containing number of stars and beta parameters

    Returns:
    (list): containing the confidence interval
    """
    iterations = 1000
    sample_ratings = []
    size = len(df)
    new_data = df[['stars', 'new_params', 'estimated_helpfulness']].copy()
    for i in range(iterations):
        if i % 100 == 0:
            print(i)
        train = resample(new_data, n_samples = size)
        train['estimated_helpfulness'] = train.apply(lambda x: np.random.beta(x[1][0], x[1][1]), axis=1)
        sample_ratings.append(compute_rating(train))
    credible_interval = [round(pd.Series(sample_ratings).quantile(0.025), 1), round(pd.Series(sample_ratings).quantile(0.975), 1)]
    return credible_interval
