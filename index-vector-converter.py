import os
import pandas as pd
import warnings
import json
import numpy as np

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex

from redis.commands.search.query import  Query

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_index_list(client):
    return client.execute_command("FT._LIST")

def get_index_info(client, index_name):
    index = SearchIndex.from_existing(index_name, redis_client=client)
    return index.info(index.name)

def get_index_prefix(index_info):
    return index_info['index_definition'][3][0]

def get_vector_size(index_info):
    return index_info['vector_index_sz_mb']

def get_num_docs(index_info):
    return index_info['num_docs']

def drop_index(client, index_name):
    return client.execute_command(f"FT.DROPINDEX {index_name}")

def get_index_report(client):
    index_data = {
        "name": [],
        "vector_size" :[],
        "num_docs":[]
    }

    index_list = get_index_list(client)

    for index in index_list:
        index_info = get_index_info(client, index)
        vector_size = get_vector_size(index_info)
        num_docs = get_num_docs(index_info)

        index_data['name'].append(index.decode('utf-8'))
        index_data['vector_size'].append(vector_size)
        index_data['num_docs'].append(num_docs)

    #CREATE DATA FRAME
    df = pd.DataFrame(index_data)

    df_sorted = df.sort_values(by="num_docs", ascending=False)

    #SAVE AS CSV
    df_sorted.to_csv(f"./index-report.csv", index=False)


def convert_vectors(client, index_name, embedding32_name, embedding16_name, batch_size, max_keys):
    #GET KEY PREFIX OF INDEX
    index_info = get_index_info(client, index_name)
    key_prefix = get_index_prefix(index_info)
    num_docs = get_num_docs(index_info)
    vector_size = get_vector_size(index_info)
    match_pattern = f"{key_prefix}*"

    print(f"Key Prefix: {key_prefix} | Vector Size: {vector_size} | Num Docs: {num_docs}")


    cursor = 0
    keys = []

    num_keys = 0

    pipeline = client.pipeline(transaction=False)
    
    while (cursor < max_keys) or (max_keys == 0):
        cursor, batch_keys = client.scan(cursor, match=match_pattern, count=batch_size)
        process_keys(pipeline, batch_keys, embedding32_name, embedding16_name)
        num_keys = num_keys + len(batch_keys)

        if cursor == 0:
            break

        
    print(f"Total Keys Processed: {num_keys}")

    return keys

def process_keys(pipeline, key_list, embedding32_name, embedding16_name):
    for key in key_list:
        pipeline.hget(key, embedding32_name)

    results = pipeline.execute()

    k = 0

    for embedding in results:
        #print(f"{key_list[k]} - {title}")
        try:
            #CONVERT TO float16
            pipeline.hset(key_list[k], embedding16_name, np.frombuffer(embedding, dtype=np.float32).astype(np.float16).tobytes())
        except:
            print(f"Error for key {key_list[k]}")
        
        k = k + 1

    pipeline.execute(True)

    return

def delete_embeddings(client, index_name, embedding_field):
    #GET KEY PREFIX OF INDEX
    index_info = get_index_info(client, index_name)
    key_prefix = get_index_prefix(index_info)
    num_docs = get_num_docs(index_info)
    match_pattern = f"{key_prefix}*"

    print(f"Key Prefix: {key_prefix} |  Num Docs: {num_docs}")

    cursor = 0
    keys = []

    num_keys = 0

    pipeline = client.pipeline(transaction=False)
    
    while True:
        cursor, batch_keys = client.scan(cursor, match=match_pattern, count=5000)
        for key in batch_keys:
            pipeline.hdel(key, embedding_field)

        pipeline.execute()
        
        num_keys = num_keys + len(batch_keys)

        if cursor == 0:
            break

        
    print(f"Total Keys Processed: {num_keys}")

    return keys




def main():
    warnings.filterwarnings('ignore')
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    client = Redis.from_url(REDIS_URL)

    print(f"Test Connection {client.ping()}")

    option = input("[1] Get Index Report\n[2] Transform Index\n[3] Delete Embedding\n[4] Create Index\n[5] Drop Index\nSelect: ")

    if option == "1":
        get_index_report(client)

    elif option == "2":
        index_name = input("Enter the Index Name: ")
        batch_size = 5000
        max_keys = 0
        convert_vectors(client, index_name, "embedding", "embedding16", batch_size, max_keys)
    elif option == "3":
        index_name = input("Enter the Index Name: ")
        embedding_field  = input("Enter the Embedding Field Name: ")
        delete_embeddings(client, index_name, embedding_field)

    elif option == "4":
        #CREATE FLOAT16 INDEX
        print("Creating new Index")
        index_def = load_dict_from_file("./movie-index-hash-2.json")
        movie_schema = IndexSchema.from_dict(index_def)
        index = SearchIndex(movie_schema, redis_client=client)
        index.create(overwrite=True, drop=False)

    elif option == "5":
        index_name = input("Enter the Index Name: ")
        drop_index(client, index_name)


if __name__ == "__main__":
    main()