# %% [markdown]
# <i>Copyright (c) Recommenders contributors.</i>
# 
# <i>Licensed under the MIT License.</i>

# %% [markdown]
# # # TF-IDF Content-Based Recommendation on Movielens Dataset
# #
# This notebook demonstrates a content-based recommendation system using the TF-IDF (Term Frequency-Inverse Document Frequency) technique on the Movielens dataset. The system recommends movies to users based on the similarity of their liked movie genres.

# %%
import sys

from recommenders.datasets import movielens
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# %% [markdown]
# ### Variable for prediction
# 
# These variables define the number of top recommendations to generate (`TOP_K`), the specific user to generate recommendations for (`user_id`), and the size of the Movielens dataset to use (`MOVIELENS_DATA_SIZE`).

# %%
# Top k items to recommend
TOP_K = 10
# User ID to recommend for
user_id = 164
# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"
# Load movies dataset into a dataframe

# %% [markdown]
# ### 1. Load the dataset into a dataframe
# Let's begin by loading the metadata file for the dataset into a Pandas dataframe. We load dataframe for user and items (movies).
# 
# This cell loads the movie metadata (movie ID, title, genres) into a Pandas DataFrame using the `movielens.load_item_df()` function from the Recommenders library.

# %%
item_data = movielens.load_item_df(
    size=MOVIELENS_DATA_SIZE,
    local_cache_path=None,
    movie_col="movieID",
    title_col="title",
    genres_col="genres"
)

# item_data.head()

# %% [markdown]
# This cell loads the user interaction data (user ID, item ID, rating) into another Pandas DataFrame.

# %%
# Load user interaction data
user_data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE, header=["userID", "itemID", "rating"]
)
# user_data.head()

# %% [markdown]
# The user interaction data is then split into training and testing sets using `python_random_split` to evaluate the model's performance on unseen data.

# %%
# Split user data into train and test sets
train_user_data, test_user_data = python_random_split(user_data, 0.75)

# %% [markdown]
# ### 2. Initialize Model
# Here, we initialize the `TfidfRecommender` model. The `tokenization_method="none"` indicates that we'll treat the genres string as a single document for TF-IDF. The `clean_dataframe` function preprocesses the 'genres' column.

# %%
recommender = TfidfRecommender(
    id_col='movieID',
    tokenization_method="none"
) # Create the recommender object
cols_to_clean = ['genres']
clean_col = 'cleaned_genres'
item_data_clean = recommender.clean_dataframe(
    item_data.copy(),
    cols_to_clean, clean_col
) # Clean dataframe

# %% [markdown]
# The `tokenize_text` function calculates the TF-IDF matrix based on the cleaned genres. The `fit` function trains the TF-IDF vectorizer.

# %%
tf, vectors_tokenized = recommender.tokenize_text(
    item_data_clean,
    text_col=clean_col
) # Tokenize text
# Fit the TF-IDF vectorizer (calculate TF-IDF matrix)
recommender.fit(tf, vectors_tokenized)

# %% [markdown]
# ### 3. Recommend Movies
# This function `recommend_movies_for_user` takes a list of liked movie IDs, seen movie IDs, the movie DataFrame, and the TF-IDF matrix as input. It calculates a user profile based on the average TF-IDF vector of the liked movies and then finds the top `total_recommend` movies with the highest cosine similarity to this user profile, excluding the movies the user has already seen. Recommendations are grouped by similarity for diversity.

# %%
# Recommend movies for a user
def recommend_movies_for_user(movie_ids_user_likes, movies_user_seen, df_movies, tfidf_matrix, total_recommend=20):

    if not movie_ids_user_likes:
        return None
    n_per_group=total_recommend // 4

    # Map movieID → index
    movie_id_to_index = pd.Series(df_movies.index.values, index=df_movies['movieID']).to_dict()
    liked_indexes = [movie_id_to_index[mid] for mid in movie_ids_user_likes if mid in movie_id_to_index]

    if not liked_indexes:
        return None

    # Calculate user's vector
    liked_vectors = tfidf_matrix[liked_indexes]
    user_profile = normalize(np.asarray(liked_vectors.mean(axis=0)))

    # Calculate cosine similarity
    cos_sim = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Drop watched movies
    already_seen = set(movies_user_seen)
    candidate_indices = [i for i in range(len(cos_sim)) if df_movies['movieID'].iloc[i] not in already_seen]

    if not candidate_indices:
        return None

    # Group recommendations by similarity
    sim_df = df_movies.iloc[candidate_indices][['movieID', 'title', 'genres']].copy() 
    sim_df['similarity'] = cos_sim[candidate_indices]
    sim_df['sim_group'] = sim_df['similarity'].round(2)

    # Descending order
    sim_df = sim_df.sort_values(by='similarity', ascending=False)
    grouped = sim_df.groupby('sim_group', sort=False)
    sorted_groups = sorted(grouped.groups.keys(), reverse=True)

    # Get top n_per_group recommendations from each group
    final_recs = []
    for group in sorted_groups:
        group_df = grouped.get_group(group)
        final_recs.extend(group_df.head(n_per_group).to_dict(orient='records'))
        if len(final_recs) >= total_recommend:
            break

    # Return recommendations
    recommended = pd.DataFrame(final_recs).head(total_recommend).reset_index(drop=True)
    recommended = recommended.drop(columns=['sim_group'])
    return recommended


# %% [markdown]
# The `evaluate_tfidf` function iterates through each unique user in the test set, retrieves their liked and seen items from the training data, generates recommendations using `recommend_movies_for_user`, and then evaluates the recommendation quality using metrics like MAP@K, NDCG@K, Precision@K, and Recall@K. The similarity score from the cosine similarity is used as the prediction score.

# %%
# Evaluate the model
def evaluate_tfidf(test_data, item_df, tfidf_rec, top_k=10):
    all_predictions = []
    for user_id in test_data['userID'].unique():
        user_history = test_data[test_data['userID'] == user_id]
        liked_items = user_history[user_history['rating'] > 3]['itemID'].tolist()
        seen_items = user_history['itemID'].tolist()

        if liked_items:
            recommendations_df = recommend_movies_for_user(
                liked_items,
                seen_items,
                item_df,
                tfidf_rec.tfidf_matrix,
                total_recommend=top_k,
            )
            if recommendations_df is not None and not recommendations_df.empty:
                for index, row in recommendations_df.iterrows():
                    all_predictions.append({
                        'userID': user_id,
                        'itemID': row['movieID'],
                        'prediction': row['similarity'] # Using similarity as prediction score
                    })

    predictions_df = pd.DataFrame(all_predictions)
    # Prepare the predictions for evaluation
    ground_truth = test_data[['userID', 'itemID']].rename(columns={'itemID': 'true_item'})
    merged_df = pd.merge(predictions_df, ground_truth, on='userID', how='left')
    merged_df = merged_df.dropna(subset=['true_item'])

    if not merged_df.empty:
        eval_map = map_at_k(test_data, predictions_df, col_prediction="prediction", k=top_k, col_user="userID", col_item="itemID")
        eval_ndcg = ndcg_at_k(test_data, predictions_df, col_prediction="prediction", k=top_k, col_user="userID", col_item="itemID")
        eval_precision = precision_at_k(test_data, predictions_df, col_prediction="prediction", k=top_k, col_user="userID", col_item="itemID")
        eval_recall = recall_at_k(test_data, predictions_df, col_prediction="prediction", k=top_k, col_user="userID", col_item="itemID")

        print("\n--- Evaluate TF-IDF Model ---")
        print(f"MAP@{top_k}:\t\t{eval_map:.4f}")
        print(f"NDCG@{top_k}:\t\t{eval_ndcg:.4f}")
        print(f"Precision@{top_k}:\t{eval_precision:.4f}")
        print(f"Recall@{top_k}:\t\t{eval_recall:.4f}")
    else:
        print("Not enough prediction.")


# %% [markdown]
# ### 4. Test and Evaluate
# This cell demonstrates how to get recommendations for a specific `user_id` (164 in this case). It retrieves the user's liked and seen movies from the test set and then calls `recommend_movies_for_user` to get the top recommendations.

# %%
# Predict for a specific user
user_id = 164
# Get the user history from the test set
user_history = test_user_data[test_user_data['userID'] == user_id]
liked_items = user_history[user_history['rating'] > 3]['itemID'].tolist()
seen_items = user_history['itemID'].tolist()
# Recommend movies for the user
recommendations_df = recommend_movies_for_user(
    liked_items,
    seen_items,
    item_data_clean,
    recommender.tfidf_matrix,
    total_recommend=TOP_K,
)

recommendations_df

# %% [markdown]
# Finally, the `evaluate_tfidf` function is called to assess the overall performance of the TF-IDF model on the test data.

# %%
# Evaluate the model
evaluate_tfidf(test_user_data, item_data_clean, recommender, top_k=TOP_K)


