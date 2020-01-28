# IMDB-Weighted-Reviews

IMDB's scoring algorithm is a secret but seems to be very closely correlated with the average score. This means that the IMDB score is too sensitive to bots, "bought" reviews, and people who rate for political/personal reasons outside of the contents of the film. In fact, there are many websites that provide this kind of service where you can pay them to generate fake reviews.

The goal of this project is to get a more "accurate" score by giving more weight to helpful reviews. Each IMDB review has, on the bottom, "x out of y found this helpful." My thought is that the more helpful a review is, the less likely that it is fake. After all, writing a helpful review takes time and effort someone who is spamming reviews will not be able to put in this effort.

scrape_list.py is a script which gets the names and IMDB codes for each movie in IMDB's Top 1000 movies

scrape_movie_helpfulness.py then takes this list and scrapes the reviews for each movie and records the number of stars the review gave as well as the helpfulness.

compute_confidence_intervals.py then takes the dataframes from scrape_movie_helpfulness.py and computes a weighted ranking as well as a "confidence interval." I decided to implement a confidence interval because some movies have very few reviews so their ratings are not very accurate.

I used some Bayesian methods to generate the movie rankings and the confidence intervals. To predict movie helpfulness, I do a binomial regression where the feature is the number of stars and the dependent variable is the % helpfulness of the review. I use binomial because the outcomes are in [0, 1]. The outcome of the regression gives me the prediction of the mean helpfulness of the review given the number of stars. I use these mean predictions to generate a beta prior for each review. From here, we update the posterior for each review based on the number of poeple who found the review helpful.

I decided to do a Bayesian method because I needed a way to generate a "confidence interval" for my score. To get the confidence interval, I bootstrap on the data and then sample from each beta posterior. Then I take a 95% interval centered around the mean. 

The results are in final_ratings.csv
