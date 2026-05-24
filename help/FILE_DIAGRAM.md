# Project File Diagram and Notes

## 1. Overview
- Project: Movie Recommendation System (MovieLens ml-latest-small)
- Goal: build, evaluate, and deploy multi-model recommendation via Streamlit
- Models: UserCF, ItemCF, SVD, Content-Based

## 2. Core Files

### `app.py`
- Web app layer (Streamlit)
- Loads models and metadata from `models/`
- Sidebar allows choosing user, model, k, threshold, compare all
- Supports: UserCF, ItemCF (with confidence badges), SVD, Content-Based, Compare All
- Metrics display: RMSE, MAE, Precision@10, Recall@10
- Functions:
  - `load_models()`
  - `predict_usercf`, `predict_itemcf`, `predict_svd`, `predict_contentbased`
  - `get_itemcf_details`
  - `build_content_features`, `recommend_contentbased`
  - `recommend_movies`

### `Train.ipynb`
- Dataset preprocessing and split
- Trains and evaluates all models
- Saves
  - `models/usercf_model.pkl`
  - `models/itemcf_model.pkl`
  - `models/svd_model.pkl`
  - `models/content_model.pkl`
  - `models/model_metadata.pkl`
  - `models/movies_metadata_for_testing.csv`
- Includes overfitting analysis and parquet optimization

### `Data_Processing.ipynb`
- Initial cleaning/transformation of MovieLens data
- Adds parquet export for performance (e.g., `movies_metadata.parquet`)

### Notebooks (helper)
- `Test_Model.ipynb`: ad-hoc model validation

## 3. Data
- `ratings.csv`, `movies.csv`, etc. from MovieLens
- `model_features.csv`: precomputed content features
- `movies_metadata_for_testing.csv`: metadata used in app

## 4. Model Outputs
- `models/model_metadata.pkl`:
  - `user_to_idx`, `movie_to_idx`, `model mappings`, metrics (`rmse_scores`,`mae_scores`,`precision_at_10`,`recall_at_10`)
- `models/*_model.pkl` for each algorithm

## 5. New content-based integration
- `Train.ipynb`: created `content_model` with:
  - `movie_vec` (movie vector by averaged features)
  - `user_profile` aggregated from user-rated movies
- `app.py`: `predict_contentbased` uses cosine similarity and scaling to rating range

## 6. Troubleshooting
1. Run training notebook first
2. Ensure `.venv` active and dependencies installed
3. Run `streamlit run app.py`
4. Update model metadata by re-running training after data changes

## 7. How to use
- Use `QUICK_START.txt` and `HOW_TO_RUN_WEB.md` for step-by-step guidance
