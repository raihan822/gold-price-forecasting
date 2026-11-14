# Gold Returns Predictive Analytics & Clustering

This repository contains the machine learning final exam project analyzing, predicting, and clustering financial market data using a highly dimensional gold dataset.

## 📌 Project Overview
This project evaluates a dataset containing **121 feature dimensions** of historical financial indicators (interest rates, bond yields, index performance, commodity prices, etc.) to predict 22-day forward gold returns (`Gold_T+22`). The pipeline implements both Supervised Regression models and Unsupervised Clustering techniques to analyze market dynamics.

## 📊 Dataset Specifications
- **Target Variable:** `Gold_T+22` (Continuous numerical values)
- **Problem Type:** Regression (Supervised) & Clustering (Unsupervised)
- **Data Points:** 2,558 rows
- **Features:** 121 (Entirely quantitative continuous features, 0 categorical variables)
- **Data Quality:** Zero missing values and zero duplicate rows.

## 🛠️ Machine Learning Pipeline

Our custom-built preprocessing and feature engineering pipelines prevent **data leakage** and handle high multicollinearity (20% of features show strong correlations with $|r| \geq 0.95$):

1. **Data Preprocessing ColumnTransformer:**
   - **Normal Features:** Handled via Z-Score Outlier Clipping ($
	ext{threshold}=3.0$) and Standard Scaling.
   - **Skewed Features:** Identified dynamically via skewness ($>1.0$) or kurtosis ($>3.5$) thresholds, transformed using `PowerTransformer` (Yeo-Johnson) for normalization.
2. **Feature Engineering & Selection:**
   - Supervised models use `SelectKBest` with Mutual Information scores coupled with `PCA` for collinearity mitigation.
   - Chronological split via `TimeSeriesSplit` to prevent look-ahead bias on time-series historical data.

## 🤖 Models & Performance Comparison

### 1. Supervised Learning Results

| Metric | Linear Regression Pipeline (with KBest + PCA) | Random Forest Regressor Pipeline |
| :--- | :--- | :--- |
| **R² Score** (Higher is Better) | 0.0013 | **0.3769** |
| **RMSE Loss** (Lower is Better) | 0.0463 | **0.0365** |

* **Verdict:** **Random Forest Regressor** outperformed Linear Regression significantly. Financial data is highly non-linear and noisy; the ensemble architecture of Random Forest successfully mapped complex feature interactions that a linear model could not grasp.

### 2. Unsupervised Learning Results
- **Algorithm:** K-Means Clustering
- **Hyperparameter Tuning:** Silhouette scoring and Elbow Method over varying $k$.
- **Result:** Optimal $k=2$ with a maximum Silhouette score of $\approx 0.114$.
- **Analysis:** The extremely low silhouette score indicates heavily overlapping structures, suggesting the feature space lacks distinct spherical clusters naturally aligned with future gold outcomes.

## 🧠 Reflective Takeaways
- **Hardest Concept:** **Principal Component Analysis (PCA)**. Transforming high-dimensional data into orthogonal components preserving maximal variance while resolving the curse of dimensionality.
- **Easiest/Most Robust Concept:** **Random Forest**. Implementing the "wisdom of the crowd" via bootstrap aggregating (bagging) to control overfitting and manage complex regression rules.

## 📂 Getting Started
To run the notebook locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gold-returns-ml.git
   ```
2. Install requirements:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```
3. Open and execute the notebook in Google Colab or Jupyter.
```
