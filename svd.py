# %% [markdown]
# <i>Copyright (c) Recommenders contributors.</i>
# 
# <i>Licensed under the MIT License.</i>

# %% [markdown]
# # Surprise Singular Value Decomposition (SVD)
# 
# This notebook implements the Singular Value Decomposition (SVD) algorithm, a popular collaborative filtering technique. SVD was notably successful during the Netflix Prize competition, used by the winning BellKor team. It aims to discover latent factors that explain user-item interactions (ratings).

# %%
import sys
import surprise

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    get_top_k_items,
)
from recommenders.models.surprise.surprise_utils import (
    predict,
    compute_ranking_predictions,
)
from recommenders.utils.notebook_utils import store_metadata

# %% [markdown]
# This is key variables for the recommendation process, including the number of top items to recommend (`TOP_K`), a specific `user_id` for demonstration, and the size of the Movielens dataset to be used (`MOVIELENS_DATA_SIZE`).

# %%
# Top k items to recommend
TOP_K = 10
# User ID to recommend for
user_id = 164
# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"

# %% [markdown]
# ### 1. Load the dataset into a dataframe
# This cell loads the Movielens dataset into a Pandas DataFrame using the `movielens.load_pandas_df()` function. The dataset contains user-item-rating triplets.

# %%
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating"],
    local_cache_path=None
)
# data.head()

# %% [markdown]
# The loaded data is then split into training and testing sets using `python_random_split` from the Recommenders library. The training set is further converted into a Surprise `Trainset` object, which is required by the Surprise library's algorithms.

# %%
# Prepare the data for training
train, test = python_random_split(data, 0.75)
train_set = surprise.Dataset.load_from_df(
    train, reader=surprise.Reader("ml-100k")
).build_full_trainset()

# %% [markdown]
# ### 2. Initialize Model
# Here, an SVD model from the Surprise library is initialized with specific parameters: `random_state` for reproducibility, `n_factors` controlling the dimensionality of the latent factor space, `n_epochs` specifying the number of training iterations, and `verbose` to display training progress.
# The model is then trained on the `train_set`, and the training time is recorded using the `Timer` utility.

# %%
svd = surprise.SVD(
    random_state=0, n_factors=200,
    n_epochs=30, verbose=True)
with Timer() as train_time:
    svd.fit(train_set)

print(f"Took {train_time.interval} seconds for training.")

# %% [markdown]
# ### 3. Prediction for user
# This section generates predictions on the `test` set using the trained SVD model. It also computes ranking predictions for all items for the users in the training set, excluding items they have already interacted with. This is used for evaluating ranking metrics later. The prediction time is also recorded.

# %%
# Predictions on the test set
predictions = predict(
    svd, test, usercol="userID",
    itemcol="itemID"
)
# Remove seen items from the test set
with Timer() as test_time:
    all_predictions = compute_ranking_predictions(
        svd, train, usercol="userID",
        itemcol="itemID", remove_seen=True
    )
print(f"Took {test_time.interval} seconds for predictions.")

# %% [markdown]
# This cell demonstrates how to retrieve the top `TOP_K` (in this case, 10) recommended items for a specific `user_id` (164) based on the ranking predictions generated earlier.

# %%
get_top_k_items(
    all_predictions[all_predictions["userID"] == user_id],
    col_rating="prediction", k=TOP_K
)

# %% [markdown]
# ### 4. Evaluate Model
# This part evaluates the performance of the trained SVD model using various regression metrics (RMSE, MAE, R-squared, Explained Variance) on the `predictions` DataFrame and ranking metrics (MAP@K, NDCG@K, Precision@K, Recall@K) on the `all_predictions` DataFrame. These metrics quantify the accuracy and ranking quality of the recommendations.

# %%
# Evaluation metrics   
eval_rmse = rmse(test, predictions)
eval_mae = mae(test, predictions)
eval_rsquared = rsquared(test, predictions)
eval_exp_var = exp_var(test, predictions)

eval_map = map_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)
eval_precision = precision_at_k(
    test, all_predictions, col_prediction="prediction", k=TOP_K
)
eval_recall = recall_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)


print(
    "RMSE:\t\t%f" % eval_rmse,
    "MAE:\t\t%f" % eval_mae,
    "rsquared:\t%f" % eval_rsquared,
    "exp var:\t%f" % eval_exp_var,
    sep="\n",
)
print("----")
print(
    "MAP:\t\t%f" % eval_map,
    "NDCG:\t\t%f" % eval_ndcg,
    "Precision@K:\t%f" % eval_precision,
    "Recall@K:\t%f" % eval_recall,
    sep="\n",
)

# %% [markdown]
# ### 5. Baseline SVD
# This section trains a baseline SVD model using the default parameters of the Surprise library. This allows for a comparison of performance against the custom SVD model trained earlier with specific hyperparameters. The training time for the baseline model is also recorded.

# %%
# Normal SVD model
svd_baseline = surprise.SVD(random_state=0)
with Timer() as train_time_baseline:
    svd_baseline.fit(train_set)

print(f"Baseline SVD model took {train_time_baseline.interval} seconds for training.")

# %% [markdown]
# Similar to the custom SVD model, this cell generates predictions on the test set and computes ranking predictions for the baseline SVD model. The prediction time for the baseline model is also recorded.

# %%
# Predictions with the baseline model
predictions_baseline = predict(
    svd_baseline, test, usercol="userID",
    itemcol="itemID"
)
with Timer() as test_time_baseline:
    all_predictions_baseline = compute_ranking_predictions(
        svd_baseline, train, usercol="userID",
        itemcol="itemID", remove_seen=True
    )
print(f"Baseline SVD model took {test_time_baseline.interval} seconds for prediction.")

# %% [markdown]
# The performance of the baseline SVD model is evaluated using the same regression and ranking metrics as the custom model. This provides a direct comparison of their effectiveness.

# %%
# Evaluation metrics for the baseline model
eval_rmse_baseline = rmse(test, predictions_baseline)
eval_mae_baseline = mae(test, predictions_baseline)
eval_rsquared_baseline = rsquared(test, predictions_baseline)
eval_exp_var_baseline = exp_var(test, predictions_baseline)

eval_map_baseline = map_at_k(test, all_predictions_baseline, col_prediction="prediction", k=TOP_K)
eval_ndcg_baseline = ndcg_at_k(test, all_predictions_baseline, col_prediction="prediction", k=TOP_K)
eval_precision_baseline = precision_at_k(
    test, all_predictions_baseline, col_prediction="prediction", k=TOP_K
)
eval_recall_baseline = recall_at_k(test, all_predictions_baseline, col_prediction="prediction", k=TOP_K)

print("\n--- Baseline SVD Model ---")
print(
    "RMSE:\t\t%f" % eval_rmse_baseline,
    "MAE:\t\t%f" % eval_mae_baseline,
    "rsquared:\t%f" % eval_rsquared_baseline,
    "exp var:\t%f" % eval_exp_var_baseline,
    sep="\n",
)
print("----")
print(
    "MAP:\t\t%f" % eval_map_baseline,
    "NDCG:\t\t%f" % eval_ndcg_baseline,
    "Precision@K:\t%f" % eval_precision_baseline,
    "Recall@K:\t%f" % eval_recall_baseline,
    sep="\n",
)

# %% [markdown]
# ### 6. Compare baseline SVD and custom SVD
# Finally, this section prints a concise comparison of the evaluation metrics for both the custom SVD model and the baseline SVD model, allowing for easy interpretation of the impact of the chosen hyperparameters on the model's performance.

# %%
print("\n--- Compare custom and baseline ---")
print("Custom SVD:")
print(f"  RMSE: {eval_rmse:.4f}, MAE: {eval_mae:.4f}, MAP@{TOP_K}: {eval_map:.4f}, NDCG@{TOP_K}: {eval_ndcg:.4f}")
print("Baseline SVD:")
print(f"  RMSE: {eval_rmse_baseline:.4f}, MAE: {eval_mae_baseline:.4f}, MAP@{TOP_K}: {eval_map_baseline:.4f}, NDCG@{TOP_K}: {eval_ndcg_baseline:.4f}")


