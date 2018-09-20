import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def main():
    ratings_data = pd.read_csv('ratings.csv')
    movie_names = pd.read_csv('movies.csv')
    movie_data = pd.merge(ratings_data, movie_names, on='movieId')
    print(movie_data.head())
    print(movie_data.groupby('title')['rating'].mean().head())
    print(movie_data.groupby('title')['rating', 'timestamp'].mean().sort_values(ascending=False, by='rating').head())
    print(movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head())
    ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())
    print(ratings_mean_count.head())


if __name__ == '__main__':
    main()
