# Recommender System with TF-IDF and SVD on Movielens Dataset

This project implements two popular recommendation algorithms, TF-IDF (Term Frequency-Inverse Document Frequency) and Singular Value Decomposition (SVD), on the Movielens dataset. It is built upon the [Recommenders 1.2.1](https://github.com/recommenders-team/recommenders/releases/tag/1.2.1) project, with some unnecessary files removed for a focused implementation.

## Overview

This repository contains two main models for movie recommendations:

1.  **TF-IDF Content-Based Recommendation:** This approach leverages the textual content of movie genres to find similarities between movies. It builds a user profile based on the genres of movies a user has liked and recommends movies with similar genre profiles.

2.  **Surprise Singular Value Decomposition (SVD):** This is a collaborative filtering technique that decomposes the user-item interaction matrix (ratings) into lower-dimensional latent factors. It identifies underlying patterns in user preferences and item characteristics to predict future ratings and generate recommendations.

## Models Explained

### 1. TF-IDF Content-Based Recommendation

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure used to evaluate the importance of a word in a document within a collection of documents (corpus). In our case:

* **Term Frequency (TF):** The frequency of a genre within a movie's genre list.
* **Inverse Document Frequency (IDF):** A measure of how rare a genre is across all movies. Genres that appear in many movies will have a lower IDF.

The TF-IDF score reflects how unique and important a genre is to a particular movie. By calculating TF-IDF vectors for each movie and then creating a user profile based on the average TF-IDF vector of liked movies, we can use cosine similarity to find other movies with similar genre profiles to recommend.

**Implementation:** The implementation for the TF-IDF model can be found in `tfidf_movielens.ipynb`.

### 2. Surprise Singular Value Decomposition (SVD)

**Singular Value Decomposition (SVD)** is a matrix factorization technique. In the context of recommender systems, it decomposes the user-item rating matrix \(R\) into three matrices:

\[
R \approx U \Sigma V^T
\]

Where:

* \(U\) is the user-latent factor matrix.
* \(\Sigma\) is a diagonal matrix of singular values, representing the strength of each latent factor.
* \(V^T\) is the item-latent factor matrix (transpose).

By learning these latent factors, SVD can predict the rating a user might give to an item they haven't interacted with. The Surprise library provides an efficient implementation of the SVD algorithm.

**Implementation:** The implementation for the SVD model can be found in `svd_movielens.ipynb`.

## Results

The evaluation results for both the TF-IDF and SVD models can be seen directly by running the corresponding Jupyter Notebooks (`tfidf_movielens.ipynb` and `svd_movielens.ipynb`). The notebooks will print the evaluation metrics (e.g., RMSE, MAE, MAP@K, NDCG@K, Precision@K, Recall@K) after the model evaluation sections.

**To see the results:**

1.  Open the `tfidf_movielens.ipynb` notebook and run all cells. The evaluation metrics for the TF-IDF model will be printed in the output of the evaluation cell.
2.  Open the `svd_movielens.ipynb` notebook and run all cells. The evaluation metrics for both the custom and baseline SVD models will be printed in the output of their respective evaluation cells.

**Interpretation of Results:** *(You can still keep a section for interpreting what the typical values of these metrics mean for recommendation systems)*

For example:

> Generally, lower RMSE and MAE values indicate better accuracy in rating predictions. For ranking metrics like MAP@K, NDCG@K, Precision@K, and Recall@K, higher values (closer to 1) indicate better performance in recommending relevant items in the top-K list. Compare the metrics obtained from the TF-IDF and SVD models to understand their relative strengths and weaknesses on this dataset.

> The SVD model generally shows better performance in terms of RMSE and MAE, indicating more accurate rating predictions compared to the TF-IDF model. However, the ranking metrics (MAP@K, NDCG@K, Precision@K, Recall@K) might show different strengths for each model, reflecting their ability to recommend relevant items in the top-K list. The content-based TF-IDF model might excel in recommending novel items based on genre similarity, while the collaborative SVD model leverages user-item interaction patterns.

## Getting Started

To run the notebooks in this project, you will need to have Python and the necessary libraries installed. It is recommended to set up a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You might need to create a `requirements.txt` file listing the dependencies like `pandas`, `numpy`, `surprise`, `recommenders`)*

3.  **Run the Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```
    This will open the Jupyter Notebook interface in your web browser, where you can navigate to and run `tfidf_movielens.ipynb` and `svd_movielens.ipynb`.

## Project Structure