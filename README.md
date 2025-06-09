## US Chronic Disease Indicators: Mortality & Incidence Analysis

This project analyzes the **U.S. Chronic Disease Indicators dataset** with a focus on **cancer and cardiovascular disease**. It explores national health trends and demographic patterns, and builds predictive models to estimate mortality/incidence rates using Random Forest and XGBoost regressors.

### Key Features

* Data cleaning and preprocessing of over 1 million rows
* Focused analysis on select high-impact health questions
* Visualizations of trends over time and across demographics
* Predictive modeling using:

  * Random Forest
  * XGBoost (with GridSearch for optimization)
* Feature importance visualization

### Files

* `notebook.ipynb` – Main Jupyter Notebook with code and analysis
* `dashboard.py` – Optional Streamlit dashboard (if applicable)

### Note

> The dataset (`U.S._Chronic_Disease_Indicators.csv`) is **too large to upload to this repository**.
> You can download it directly from [Kaggle](https://www.kaggle.com/datasets/sahirmaharajj/u-s-chronic-disease-indicators) and place it in the same directory as the notebook.

### Libraries Used

* pandas
* seaborn & matplotlib
* scikit-learn
* XGBoost
* Streamlit (optional dashboard)
