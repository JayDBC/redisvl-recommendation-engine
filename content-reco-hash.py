import ast
import os
import pandas as pd
import pickle
import requests
import warnings
import json
import numpy as np

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.query.filter import Tag, Num, Text

from redis.commands.search.query import  Query


vec_count = 0

def prepare_data(data_type, vectorizer, num_dim):

    df = pd.read_csv("data/25k_imdb_movie_dataset.csv", nrows=20000)
    df.head()

    df.drop(columns=['runtime', 'genres', 'keywords','cast', 'writer', 'path'], inplace=True)


    # make sure we've filled all missing values
    df.isnull().sum()

    # add a column to the dataframe with all the text we want to embed
    df["full_text"] = df["title"] + ". " + df["overview"]


    try:
        #LOAD CACHED VECTOR EMBEDDINGS
        print(f'Loading data/embeddings_{data_type}_{num_dim}.pkl')
        with open(f'data/embeddings_{data_type}_{num_dim}.pkl', 'rb') as vector_file:
            df['embedding'] = pickle.load(vector_file)         
    except:
        #VECTORIZE DATA
        print("Could not load cached Vectors...")
        df['embedding'] = df['full_text'].apply(lambda x: create_embedding(x, vectorizer, data_type))
        #Save the data with embeddings
        pickle.dump(df['embedding'], open(f'data/embeddings_{data_type}_{num_dim}.pkl', 'wb'))
        


    return df


def create_embedding(x, vectorizer, data_type):
    global vec_count
    if(vec_count % 1000 == 0):
        print(f"Vectorized {vec_count}")
    embedding = vectorizer.embed(f"{x}", as_buffer=False)
    vec_count = vec_count + 1
    return np.array(embedding).astype(data_type).tobytes()


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

def get_recommendations(index, movie_vector, data_type='float32', filter=None, num_results=5, distance=0.6):
    query = RangeQuery(
        vector=movie_vector,
        vector_field_name='embedding',
        dtype=data_type,
        num_results=num_results,
        distance_threshold=distance,
        return_fields=['title', 'full_text'],
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
    qry.paging(0, 1)
    qry.return_field("embedding", decode_field=False)
    qry.return_fields("title","year")
    res = rs.search(qry)
    return res





def main():
    warnings.filterwarnings('ignore')
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    REDIS_URL = f"redis://:{REDIS_PASSWORD}@redis-12400.jsd-aa1.demo.redislabs.com:12400"


    client = Redis.from_url(REDIS_URL)
    print(f"Test Connection {client.ping()}")


    #load the index definition
    index_def = load_dict_from_file("./movie-index-hash.json")
    movie_schema = IndexSchema.from_dict(index_def)
    index = SearchIndex(movie_schema, redis_client=client)

    
    #create native redis search object
    rs = client.ft(index_def['index']['name'])

    #create vectorizer
    #vectorizer = HFTextVectorizer(model='sentence-transformers/paraphrase-MiniLM-L6-v2')
    vectorizer = HFTextVectorizer(model='sentence-transformers/all-roberta-large-v1')
    #vectorizer = HFTextVectorizer()

    e1 = vectorizer.embed('romantic movies')
    num_dims = len(e1)
    print(f"Num Dims: {num_dims}")

    
    #choose the data type float16/float32
    data_type = input("What is the precision of the vector embeddings (float16/float32) ? ")

    #materialize the search index
    load_data = input("Would you like to load/reload data? (y/n) ")

    #LOAD DATA
    if load_data.startswith("y"):
        print("Preparing Data ...")
        df = prepare_data(data_type, vectorizer, num_dims)
        #CREATE INDEX
        index.create(overwrite=True, drop=True)

        #LOAD DATA
        print("Loading Data into Redis ...")
        data = df.to_dict(orient='records')
        index.load(data)


    index_list = client.execute_command("FT._LIST")

    for idx in index_list:
        print(f"{idx.decode('utf-8')} , ")


    index2 = SearchIndex.from_existing("idx:movies", redis_client=client)
    
    #print(f"{index2.info("idx:rag")}")

    mode = input("\nSelect query mode: [1] Text Search [2] Vector Search [3] Movie Recommendations : ")

    prompt_text = "What movie did you Watch ? "

    if mode != "3":
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

        elif mode == "2":
            #vectorize the search string     
            search_vector = vectorizer.embed(qry, as_buffer=False)

            if data_type == 'float16':
                print("Float16 Vectors")
                search_vector = np.array(search_vector).astype(data_type).tobytes()


            filter = None          
                
            recs = get_recommendations(index, search_vector, data_type, filter, num_results=5, distance=0.6)

            print("\nI found the following movies\n")
            for rec in recs:
                print(f">> {rec['title']}:\n\t {rec['full_text']}")

            print("-------------------------------------------------")

        else:
            #recommend similar movies
            search_result = get_movie_vector(rs, f"@title:({qry})")

            if search_result.total > 0:
                #get the movie vector, we will take the first match
                print(f"{search_result.docs[0].title}")
                movie_vector = np.frombuffer(bytearray(search_result.docs[0].embedding), "float32").tolist()
                
                filter = None
                    
                recs = get_recommendations(index, movie_vector, data_type, filter, num_results=5, distance=0.8)

                print("\nI recommend the following movies\n")
                for rec in recs:
                    print(f">> {rec['title']}: \n\t {rec['full_text']}")

                print("-------------------------------------------------")
        



if __name__ == "__main__":
    main()