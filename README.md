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

**Implementation:** The implementation for the TF-IDF model can be found in `model/tfidf_movielens.ipynb`.

### 2. Surprise Singular Value Decomposition (SVD)

**Singular Value Decomposition (SVD)** is a matrix factorization technique. In the context of recommender systems, it decomposes the user-item rating matrix \(R\) into three matrices:

$$
R \approx U \Sigma V^T
$$

Where:

* $$U$$ is the user-latent factor matrix.
* $$\Sigma$$ is a diagonal matrix of singular values, representing the strength of each latent factor.
* $$V^T$$ is the item-latent factor matrix (transpose).

By learning these latent factors, SVD can predict the rating a user might give to an item they haven't interacted with. The Surprise library provides an efficient implementation of the SVD algorithm.

**Implementation:** The implementation for the SVD model can be found in `model/svd_movielens.ipynb`.

## Results

The evaluation results for both the TF-IDF and SVD models can be seen directly by running the corresponding Jupyter Notebooks (`tfidf_movielens.ipynb` and `svd_movielens.ipynb`). The notebooks will print the evaluation metrics (e.g., RMSE, MAE, MAP@K, NDCG@K, Precision@K, Recall@K) after the model evaluation sections.

**To see the results:**

1.  Open the `tfidf_movielens.ipynb` notebook and run all cells. The evaluation metrics for the TF-IDF model will be printed in the output of the evaluation cell.
2.  Open the `svd_movielens.ipynb` notebook and run all cells. The evaluation metrics for both the custom and baseline SVD models will be printed in the output of their respective evaluation cells.

**Interpretation of Results:** 

I. The evaluation of the TF-IDF content-based recommender system yielded scores of 0 for MAP@K, NDCG@K, Precision@K, and Recall@K. This suggests that the model, in its current configuration, is not effectively recommending movies that align with the positive interactions (likes) observed in the test dataset. Several factors likely contribute to this outcome:

    1. Limitations of Content Representation (Insufficient Granularity):
    
        As you correctly pointed out, the content representation is solely based on the 'genres' of the movies. With only 19 distinct genres available in the Movielens dataset, the granularity of movie descriptions is quite coarse. Many movies share the same genre classifications, potentially leading to high similarity scores between movies that users might perceive as quite different. This lack of nuanced content features makes it challenging for the TF-IDF model to accurately capture the subtle differences in movie characteristics that drive individual user preferences. Consequently, the recommendations generated based on broad genre overlap may not resonate with specific user tastes.
        
    2.  Disconnect Between "Seen" and "Favorite" (Weak Signal in User History):
    
        Your observation that user interactions might be heavily influenced by factors other than genuine preference (e.g., popularity, trends) is crucial. If a significant portion of the movies users have seen and rated positively were driven by external factors rather than intrinsic enjoyment of the movie's content, then using these "likes" to build a genre-based user profile becomes less reliable. The model is learning preferences based on what users watched and rated favorably for various reasons, not necessarily what they truly favor in terms of underlying content. This introduces noise into the user profiles and hinders the model's ability to recommend genuinely relevant content.
        
    3. Potential Use Case: "Similar to What You've Seen" Section:
    
        Despite the poor performance in a general recommendation setting, your insight about the TF-IDF model's tendency to recommend movies with similar genres highlights a potential niche application: a "Similar to What You've Seen" section. In this context, the model's strength in identifying movies with overlapping genre classifications could be valuable for users looking for more of what they have already experienced, regardless of the underlying reasons for their initial viewing. This feature could cater to users who enjoyed a movie for its genre and are seeking similar experiences.
        
II. The evaluation of the 2 SVD model is the score of basic level recommendation quality:

    1. While the Baseline SVD exhibits slightly better performance in terms of rating prediction accuracy (lower RMSE and MAE, higher R-squared and explained variance), the Custom SVD shows a slight advantage in ranking quality, particularly as indicated by the higher NDCG@K and Recall@K. The MAP@K and Precision@K are quite similar between the two models.
    
    2. Modifications made in the Custom SVD model (SVD + ML) might have slightly shifted the model's focus towards better ranking of relevant items, potentially at the cost of a minor decrease in pure rating prediction accuracy.
    
    3. The choice of which model is "better" would depend on the specific goals of the recommender system. If accurate rating prediction is paramount, the Baseline SVD might be preferred. However, if the primary goal is to provide well-ranked and relevant recommendations, the Custom SVD appears to be slightly more effective. The differences, however, are relatively small, indicating that both the standard SVD and the customized version perform comparably on this dataset.
