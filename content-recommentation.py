import ast
import os
import pandas as pd
import pickle
import requests
import warnings
import json

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.query.filter import Tag, Num, Text


class ContentEngine:

    index
    client
    df

    def __init__(self,secret=""):
        warnings.filterwarnings('ignore')
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = os.getenv("REDIS_PORT", "6379")
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
        REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

        self.client = Redis.from_url(REDIS_URL)
        print(f"Test Connection {client.ping()}")

        #prepare data
        print("Preparing Data")
        df = prepare_data()

        #load the index definition
        movie_schema = IndexSchema.from_dict(load_dict_from_file("./movie-index.json"))
        self.index = SearchIndex(movie_schema, redis_client=client)

        self.df = prepare_data()


    def prepare_data():
        try:
            df = pd.read_csv("data/25k_imdb_movie_dataset.csv")
        except:
            import requests
            # download the file
            url = 'https://redis-ai-resources.s3.us-east-2.amazonaws.com/recommenders/datasets/content-filtering/25k_imdb_movie_dataset.csv'
            r = requests.get(url)

            #save the file as a csv
            if not os.path.exists('./data'):
                os.makedirs('./data')
            with open('./data/25k_imdb_movie_dataset.csv', 'wb') as f:
                f.write(r.content)
            df = pd.read_csv("data/25k_imdb_movie_dataset.csv")

        df.head()

        df.drop(columns=['runtime', 'writer', 'path'], inplace=True)
        df['year'] = df['year'].apply(replace_year)             # replace roman numerals with average year
        df['genres'] = df['genres'].apply(ast.literal_eval)     # convert string representation of list to list
        df['keywords'] = df['keywords'].apply(ast.literal_eval) # convert string representation of list to list
        df['cast'] = df['cast'].apply(ast.literal_eval)         # convert string representation of list to list
        df = df[~df['overview'].isnull()]                       # drop rows with missing overviews
        df = df[~df['overview'].isin(['none'])]                 # drop rows with 'none' as the overview

        # make sure we've filled all missing values
        df.isnull().sum()

        # add a column to the dataframe with all the text we want to embed
        df["full_text"] = df["title"] + ". " + df["overview"] + " " + df['keywords'].apply(lambda x: ', '.join(x))
        df["full_text"][0]

        #Vectorize the data from scratch
        #vectorizer = HFTextVectorizer(model='sentence-transformers/paraphrase-MiniLM-L6-v2')
        #df['embedding'] = df['full_text'].apply(lambda x: vectorizer.embed(x, as_buffer=False))
        #pickle.dump(df['embedding'], open('datasets/content_filtering/text_embeddings.pkl', 'wb'))

        try:
            #load pre-vectorized data
            with open('data/text_embeddings.pkl', 'rb') as vector_file:
                df['embedding'] = pickle.load(vector_file)
        except:
            embeddings_url = 'https://redis-ai-resources.s3.us-east-2.amazonaws.com/recommenders/datasets/content-filtering/text_embeddings.pkl'
            r = requests.get(embeddings_url)
            with open('./data/text_embeddings.pkl', 'wb') as f:
                f.write(r.content)
            with open('data/text_embeddings.pkl', 'rb') as vector_file:
                df['embedding'] = pickle.load(vector_file)

        return df



    def replace_year(x):
        roman_numerals = ['(I)','(II)','(III)','(IV)', '(V)', '(VI)', '(VII)', '(VIII)', '(IX)', '(XI)', '(XII)', '(XVI)', '(XIV)', '(XXXIII)', '(XVIII)', '(XIX)', '(XXVII)']
        if x in roman_numerals:
            return 1998 # the average year of the dataset
        else:
            return x
    
    def load_dict_from_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def make_filter(genres=None, release_year=None, keywords=None):
        flexible_filter = (
            (Num("year") > release_year) &  # only show movies released after this year
            (Tag("genres") == genres) &     # only show movies that match at least one in list of genres
            (Text("full_text") % keywords)  # only show movies that contain at least one of the keywords
        )
        return flexible_filter

    def get_recommendations(index, movie_vector, num_results=5, distance=0.6, filter=None):
        query = RangeQuery(
            vector=movie_vector,
            vector_field_name='embedding',
            num_results=num_results,
            distance_threshold=distance,
            return_fields=['title', 'overview', 'genres'],
            filter_expression=filter,
        )

        recommendations = index.query(query)
        return recommendations

    def get_movie_vector(df, movie_title):
        try:
            vec = df[df['title'] == movie_title]['embedding'].values[0]
            return vec
        except Exception as e:
            return "NA"




def main():
    content_engine = 
    warnings.filterwarnings('ignore')
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    client = Redis.from_url(REDIS_URL)
    print(f"Test Connection {client.ping()}")

    #prepare data
    print("Preparing Data")
    df = prepare_data()

    #load the index definition
    movie_schema = IndexSchema.from_dict(load_dict_from_file("./movie-index.json"))
    index = SearchIndex(movie_schema, redis_client=client)

    #create the search index
    load_data = input("Would you like to load/reload data? (y/n) ")

    if load_data.startswith("y"):
        index.create(overwrite=True, drop=True)
        #load the data into redis
        print("Loading Data into Redis")
        data = df.to_dict(orient='records')
        index.load(data)


    while True:
        movie = input("What movie did the user watch? ")

        if movie == "bye" or movie == "quit":
            break
        else:
            movie_vector = get_movie_vector(df, movie)
            if movie_vector != "NA":
                filter = make_filter(genres=['Horror'], release_year=1990)
                recs = get_recommendations(index, movie_vector, distance=0.8)
                result_df = {
                    "title" : [],
                    "genres" :[],
                    "overview":[]
                }
                for rec in recs:
                    result_df['title'].append(rec['title'])
                    #result_df['genres'].append(",".merge(rec['genres']))
                    result_df['overview'].append(rec['overview'])
                    #print(f"- {rec['title']}:\n\tGenres: {rec['genres']}\n\t {rec['overview']}")

                print(result_df)
        


if __name__ == "__main__":
    main()