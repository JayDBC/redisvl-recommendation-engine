# Recommendation Engine using Semantic Search.

![Redis](https://redis.io/wp-content/uploads/2024/04/Logotype.svg?auto=webp&quality=85,75&width=120)

This is a simple implementation of a Recommendation Engine for the IMDB Movie dataset leveraging semantic search.<br>

## Setup
#1
Download <i>25k_imdb_movie_dataset.csv</i> and text_embeddings.pkl using the links below and place them in ./data folder.<br>
- https://redis-ai-resources.s3.us-east-2.amazonaws.com/recommenders/datasets/content-filtering/25k_imdb_movie_dataset.csv<br>
- https://redis-ai-resources.s3.us-east-2.amazonaws.com/recommenders/datasets/content-filtering/text_embeddings.pkl<br><br>
#2
Add REDIS_HOST, REDIS_PORT, REDIS_PASSWORD as environment variables.<br>

## Execution
Execute the content-reco.py file and follow the prompts.

## Documentation

- https://www.redisvl.com/ RedisVL Library
- https://redis-py.readthedocs.io/en/stable/index.html Redis Python client docs