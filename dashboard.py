import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Streamlit page setup
st.set_page_config(page_title="Chronic Disease Indicators Dashboard", layout="wide")
st.title("📊 U.S. Chronic Disease Indicators Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')
    cols = ['YearStart', 'LocationDesc', 'Topic', 'Question', 'DataValue',
            'LowConfidenceLimit', 'HighConfidenceLimit', 'StratificationCategory1',
            'Stratification1']
    df = df[cols]
    df = df[df['DataValue'].notnull()]
    return df

df = load_data()
topics_of_interest = ['Cardiovascular Disease', 'Cancer']

# Sidebar filters
st.sidebar.header("Filters")
selected_topic = st.sidebar.selectbox("Select Topic", topics_of_interest)
questions_available = df[df['Topic'] == selected_topic]['Question'].unique()
selected_question = st.sidebar.selectbox("Select Question", sorted(questions_available))

df_filtered = df[(df['Topic'] == selected_topic) & (df['Question'] == selected_question)]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Dataset", "📈 Trends", "👥 Demographics", "🤖 Models", "📌 Feature Importances"])

# Tab 1: Dataset Overview
with tab1:
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

# Tab 2: Trends
with tab2:
    st.subheader("Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_filtered, x="YearStart", y="DataValue", hue="Stratification1", ax=ax)
    ax.set_title(f"{selected_question} Over Time by Stratification")
    st.pyplot(fig)

# Tab 3: Demographic Breakdown
with tab3:
    st.subheader("Demographic Group Comparison")
    df_demo = df_filtered[df_filtered['Stratification1'] != 'Overall']
    strat = df_demo.groupby('Stratification1')['DataValue'].mean().sort_values()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    strat.plot(kind='barh', ax=ax2)
    ax2.set_title(f"{selected_question} by Demographic Group")
    ax2.set_xlabel("Average Rate")
    st.pyplot(fig2)

# Tab 4: ML Models
with tab4:
    st.subheader("Model Training and Evaluation")

    df_model = df_filtered[df_filtered['DataValue'] < 1000].copy()
    df_model = df_model[['YearStart', 'LocationDesc', 'Stratification1', 'DataValue']].dropna()
    df_model[['YearStart']] = StandardScaler().fit_transform(df_model[['YearStart']])
    df_model = pd.get_dummies(df_model, columns=['LocationDesc', 'Stratification1'], drop_first=True)

    X = df_model.drop(columns=['DataValue'])
    y = df_model['DataValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax3)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    ax3.set_title("Predicted vs Actual")
    st.pyplot(fig3)

# Tab 5: Feature Importances
with tab5:
    st.subheader("Top 10 Feature Importances (XGBoost)")
    final_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    final_model.fit(X_train, y_train)
    importance = final_model.get_booster().get_score(importance_type='gain')
    
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_items:
        features, scores = zip(*sorted_items)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
        ax4.barh(features[::-1], scores[::-1], color=colors[::-1])
        ax4.set_xlabel("Gain Importance")
        ax4.set_title("Top 10 Feature Importances")
        st.pyplot(fig4)
    else:
        st.write("Not enough data to compute feature importances.")
