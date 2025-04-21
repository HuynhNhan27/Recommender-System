# %% [markdown]
# <i>Copyright (c) Recommenders contributors.</i>
# 
# <i>Licensed under the MIT License.</i>

# %% [markdown]
# # TF-IDF Content-Based Recommendation on the COVID-19 Open Research Dataset
# This demonstrates a simple implementation of Term Frequency Inverse Document Frequency (TF-IDF) content-based recommendation on the [COVID-19 Open Research Dataset](https://azure.microsoft.com/en-us/services/open-datasets/catalog/covid-19-open-research/), hosted through Azure Open Datasets.
# 
# In this notebook, we will create a recommender which will return the top k recommended articles similar to any article of interest (query item) in the COVID-19 Open Research Dataset.

# %%
import sys
import nltk

from recommenders.datasets import movielens
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender

nltk.download("punkt_tab")
# Print version
print(f"System version: {sys.version}")

# %%
import sys
import importlib

from recommenders.datasets import movielens
from recommenders.models.tfidf import tfidf_utils
importlib.reload(tfidf_utils)
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender
# Print version
print(f"System version: {sys.version}")

# %% [markdown]
# ### 1. Load the dataset into a dataframe
# Let's begin by loading the metadata file for the dataset into a Pandas dataframe. This file contains metadata about each of the scientific articles included in the full dataset.

# %%
# Top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE, header=["userID", "itemID", "rating"]
)

data = movielens.load_item_df(
    size=MOVIELENS_DATA_SIZE,
    local_cache_path=None,
    movie_col="movieID",
    title_col="title",
    genres_col="genres",
    year_col=None
)

data.head()

# %%
# Create the recommender object
recommender = TfidfRecommender(id_col='movieID', tokenization_method="simple")

# %%
# Assign columns to clean and combine
cols_to_clean = ['genres']
clean_col = 'cleaned_genres'
data_clean = recommender.clean_dataframe(data, cols_to_clean, clean_col)

# %%
data_clean.head()
# print(data_clean[clean_col].head().tolist())

# %%
tf, vectors_tokenized = recommender.tokenize_text(data_clean, text_col=clean_col)
vectors_tokenized.head()

# %%
# vectors_tokenized[1].lower().split()

# %%
# Fit the TF-IDF vectorizer
recommender.fit(tf, vectors_tokenized)

# %%
recommender.tfidf_matrix

# %%
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas
import numpy as np

def recommend_movies(movie_ids_user_likes, df_movies, tfidf_matrix, top_n=10):
    # Map movieId → index trong tfidf_matrix
    movie_id_to_index = pandas.Series(df_movies.index.values, index=df_movies['movieID']).to_dict()
    liked_indexes = [movie_id_to_index[mid] for mid in movie_ids_user_likes if mid in movie_id_to_index]

    if not liked_indexes:
        return pandas.DataFrame(columns=['movieID', 'title'])

    # Tính vector người dùng
    liked_vectors = tfidf_matrix[liked_indexes]
    user_profile = normalize(np.asarray(liked_vectors.mean(axis=0)))

    # Tính similarity
    cos_sim = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Bỏ qua phim đã xem
    already_seen = set(liked_indexes)
    top_indices = [i for i in cos_sim.argsort()[::-1] if i not in already_seen][:top_n]

    # Trả về DataFrame gợi ý
    recommended = df_movies.iloc[top_indices][['movieID', 'title', 'genres']]
    recommended['score'] = cos_sim[top_indices]  # thêm điểm similarity nếu cần
    return recommended.reset_index(drop=True)


recommend_movies([1, 10, 25], data_clean, recommender.tfidf_matrix, top_n=5)


# %%
# Get recommendations
top_k_recommendations = recommender.recommend_top_k_items(data_clean, k=5)

# %%
recommender.get_top_k_recommendations(data_clean, "1")

# %% [markdown]
# ### 3. Retrieve full article text
# Now that we have the metadata for the public domain articles as its own dataframe, let's retrieve the full text for each public domain scientific article.

# %%
# Extract text from all public domain articles (may take 2-3 min)
all_text = covid_utils.get_public_domain_text(df=metadata_public, container_name=container_name, azure_storage_sas_token=sas_token)

# %% [markdown]
# Notice that **all_text** is the same as **metadata_public** but now has an additional column called **full_text** which contains the full text for each respective article.

# %%
# Preview
all_text.head()

# %% [markdown]
# ### 4. Instantiate the recommender
# All functions for data preparation and recommendation are contained within the **TfidfRecommender** class we have imported. Prior to running these functions, we must create an object of this class.
# 
# Select one of the following tokenization methods to use in the model:
# 
# | tokenization_method | Description                                                                                                                      |
# |:--------------------|:---------------------------------------------------------------------------------------------------------------------------------|
# | 'none'              | No tokenization is applied. Each word is considered a token.                                                                     |
# | 'nltk'              | Simple stemming is applied using NLTK.                                                                                           |
# | 'bert'              | HuggingFace BERT word tokenization ('bert-base-cased') is applied.                                                               |
# | 'scibert'           | SciBERT word tokenization ('allenai/scibert_scivocab_cased') is applied.<br>This is recommended for scientific journal articles. |

# %%
# Create the recommender object
recommender = TfidfRecommender(id_col='cord_uid', tokenization_method='scibert')

# %% [markdown]
# ### 5. Prepare text for use in the TF-IDF recommender
# The raw text retrieved for each article requires basic cleaning prior to being used in the TF-IDF model.
# 
# Let's look at the full_text from the first article in our dataframe as an example.

# %%
# Preview the first 1000 characters of the full scientific text from one example
print(all_text['full_text'][0][:1000])

# %% [markdown]
# As seen above, there are some special characters (such as • ▲ ■ ≥ °) and punctuation which should be removed prior to using the text as input. Casing (capitalization) is preserved for [BERT-based tokenization methods](https://huggingface.co/transformers/model_doc/bert.html), but is removed for simple or no tokenization.
# 
# Let's join together the **title**, **abstract**, and **full_text** columns and clean them for future use in the TF-IDF model.

# %%
# Assign columns to clean and combine
cols_to_clean = ['title','abstract','full_text']
clean_col = 'cleaned_text'
df_clean = recommender.clean_dataframe(all_text, cols_to_clean, clean_col)

# %%
# Preview the dataframe with the cleaned text
df_clean.head()

# %%
# Preview the first 1000 characters of the cleaned version of the previous example
print(df_clean[clean_col][0][:1000])

# %% [markdown]
# Let's also tokenize the cleaned text for use in the TF-IDF model. The tokens are stored within our TfidfRecommender object.

# %%
# Tokenize text with tokenization_method specified in class instantiation
tf, vectors_tokenized = recommender.tokenize_text(df_clean, text_col=clean_col)

# %% [markdown]
# ### 6. Recommend articles using TF-IDF
# Let's now fit the recommender model to the processed data (tokens) and retrieve the top k recommended articles.
# 
# When creating our object, we specified k=5 so the `recommend_top_k_items` function will return the top 5 recommendations for each public domain article.

# %%
# Fit the TF-IDF vectorizer
recommender.fit(tf, vectors_tokenized)

# Get recommendations
top_k_recommendations = recommender.recommend_top_k_items(df_clean, k=5)

# %% [markdown]
# In our recommendation table, each row represents a single recommendation.
# 
# - **cord_uid** corresponds to the article that is being used to make recommendations from.
# - **rec_rank** contains the recommdation's rank (e.g., rank of 1 means top recommendation).
# - **rec_score** is the cosine similarity score between the query article and the recommended article.
# - **rec_cord_uid** corresponds to the recommended article.

# %%
# Preview the recommendations
top_k_recommendations

# %% [markdown]
# Optionally, we can access the full recommendation dictionary, which contains full ranked lists for each public domain article.

# %%
# Optionally view full recommendation list
full_rec_list = recommender.recommendations

article_of_interest = 'ej795nks'
print('Number of recommended articles for ' + article_of_interest + ': ' + str(len(full_rec_list[article_of_interest])))

# %% [markdown]
# Optionally, we can also view the tokens and stop words which were used in the recommender.

# %%
# Optionally view tokens
tokens = recommender.get_tokens()

# Preview 10 tokens
print(list(tokens.keys())[:10])

# %%
# Preview just the first 10 stop words sorted alphabetically
stop_words = list(recommender.get_stop_words())
stop_words.sort()
print(stop_words[:10])

# %% [markdown]
# ### 7. Display top recommendations for article of interest
# Now that we have the recommendation table containing IDs for both query and recommended articles, we can easily return the full metadata for the top k recommendations for any given article.

# %%
cols_to_keep = ['title','authors','journal','publish_time','url']
recommender.get_top_k_recommendations(metadata_public,article_of_interest,cols_to_keep)

# %% [markdown]
# ### Conclusion
# In this notebook, we have demonstrated how to create a TF-IDF recommender to recommend the top k (in this case 5) articles similar in content to an article of interest (in this example, article with `cord_uid='ej795nks'`).


