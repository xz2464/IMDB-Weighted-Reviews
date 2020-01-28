import pandas as pd
import numpy as np
from imdb import scrape_all_code

data = pd.read_csv('movie_list.csv', index_col = 0)
scrape_all_code(data['Code'])
