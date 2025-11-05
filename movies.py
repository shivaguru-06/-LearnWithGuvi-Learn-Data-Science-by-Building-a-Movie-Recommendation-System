import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

movies=pd.read_csv('movies.csv')
ratings=pd.read_csv('ratings.csv')
print(movies)
print(ratings)
print(movies.head())
print(ratings.head())
df=ratings.merge(movies, on='movieId')
print(df.head(100))
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.info()
df.describe()
print("Total Users :", df['userId'].nunique())
print("Total Movies:", df['movieId'].nunique())
plt.figure(figsize=(6,4))
sns.countplot(x='rating', data=df)
plt.title("Distribution of Ratings")
plt.show()
top_movies = df['title'].value_counts().head(10)
print(top_movies)
top_movies.plot(kind='barh', figsize=(7,4), title="Top 10 Most Rated Movies")
plt.show()

df['genres'] = df['genres'].str.split('|')
df_exploded = df.explode('genres').reset_index(drop=True)

plt.figure(figsize=(8,5))
sns.countplot(y='genres', data=df_exploded, order=df_exploded['genres'].value_counts().index[:10])
plt.title("Top 10 Movie Genres")
plt.show()


df['genres']=df['genres'].fillna('').astype('str')

df['combined_features'] = df['title'] + " " + df['genres']
df['combined_features'].head

movies = df.copy()
movies = movies.drop_duplicates(subset='title', keep='first').reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title):
    if title not in indices:
        return "Movie not found in dataset!"

    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores_indices = sim_scores.argsort()[::-1]
    sim_scores_indices = sim_scores_indices[sim_scores_indices != idx]
    top_indices = sim_scores_indices[:10]

    return movies['title'].iloc[top_indices]

get_recommendations('Heat (1995)')