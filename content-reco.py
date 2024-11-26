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

from redis.commands.search.query import  Query


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
        return int(x)
    
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

def get_recommendations(index, movie_vector, filter=None, num_results=5, distance=0.6):
    query = RangeQuery(
        vector=movie_vector,
        vector_field_name='embedding',
        num_results=num_results,
        distance_threshold=distance,
        return_fields=['title', 'full_text', 'genres'],
        filter_expression=filter,
    )

    recommendations = index.query(query)
    return recommendations

def search_movies(rs, query_str):
    qry = Query(query_str)
    qry.paging(0, 5).return_fields("title","genres","rating","year", "keywords")
    res = rs.search(qry)
    return res


def get_movie_vector(rs, query_str):
    qry = Query(query_str)
    qry.paging(0, 5).return_fields("embedding")
    res = rs.search(qry)
    return res

def get_movie_vector0(df, movie_title):
    try:
        vec = df[df['title'] == movie_title]['embedding'].values[0]
        return vec
    except Exception as e:
        return "NA"



def main():
    warnings.filterwarnings('ignore')
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    client = Redis.from_url(REDIS_URL)
    print(f"Test Connection {client.ping()}")


    #load the index definition
    index_def = load_dict_from_file("./movie-index.json")
    movie_schema = IndexSchema.from_dict(index_def)
    index = SearchIndex(movie_schema, redis_client=client)

    #create native redis search object
    rs = client.ft(index_def['index']['name'])

    #materialize the search index
    load_data = input("Would you like to load/reload data? (y/n) ")

    if load_data.startswith("y"):
        #prepare data
        print("Preparing Data ...")
        df = prepare_data()
        index.create(overwrite=True, drop=True)
        #load the data into redis
        print("Loading Data into Redis ...")
        data = df.to_dict(orient='records')
        index.load(data)

    mode = input("\nSelect query mode: [1] Text Search [2] Vector Search ")

    prompt_text = "What movie did you Watch ? "

    if mode == "1":
        prompt_text = "Enter your Movie Search: "


    while True:
        qry = input(prompt_text)

        if qry == "bye" or qry == "quit":
            break
        elif mode == "1":   
            res = search_movies(rs, qry) 
            print(f"I found {res.total} movies.\n")
            for doc in res.docs:
                print(f"{doc.title}: Rating: {doc.rating} Year: {doc.year}\n {doc.keywords}\n")
                print("---------------------------------------------------------------")

        else:
            #recommend similar movies
            search_result = get_movie_vector(rs, f"@title:({qry})")

            if search_result.total > 0:
                #get the movie vector, we will take the first match
                movie_vector = json.loads(search_result.docs[0].embedding)

                #Input Genre
                genre = input("\nWould you filter by Genre? ")
                genre_list = None

                if genre != None or genre != "":
                    genre_list = genre.split(',')

                filter = make_filter(genres=genre_list, release_year=None)
                    
                recs = get_recommendations(index, movie_vector, filter, num_results=5, distance=0.8)

                print("\nI recommend the following movies\n")
                for rec in recs:
                    print(f">> {rec['title']}: Genre: {rec['genres']}\n\t {rec['full_text']}")

                print("-------------------------------------------------")
        


if __name__ == "__main__":
    main()