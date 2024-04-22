import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
import shap
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

def main():
    st.title('Data Exploration and Machine Learning Modeling')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if st.checkbox('Show raw data'):
            st.write("Data Preview:", data)

        if st.checkbox('Select Columns to Drop', False):
            columns_to_drop = st.multiselect('Select columns to drop', data.columns)
            data = data.drop(columns=columns_to_drop)
            st.write("Updated Data Preview:", data)

        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=np.number).columns.tolist()
        st.write("Numeric Columns:", numeric_columns)
        st.write("Categorical Columns:", categorical_columns)

        if st.checkbox('Show EDA', False):
            perform_eda(data, numeric_columns, categorical_columns)

        if st.checkbox('Build Model'):
            problem_type = st.selectbox('Select Problem Type', ['Regression', 'Classification'])
            target_variable = st.selectbox('Select Target Variable', numeric_columns if problem_type == 'Regression' else categorical_columns)
            if target_variable:
                if st.button("Use AutoML"):
                    h2o.init()  # Initialize H2O when the AutoML button is clicked
                    run_h2o_automl(data, target_variable, problem_type)
                else:
                    run_models(data, numeric_columns, categorical_columns, target_variable, problem_type)

def perform_eda(data, numeric_columns, categorical_columns):
    st.subheader('Exploratory Data Analysis')
    selected_numeric_columns = st.multiselect('Select numeric columns for EDA', numeric_columns)
    if selected_numeric_columns:
        for column in selected_numeric_columns:
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    selected_categorical_columns = st.multiselect('Select categorical columns for EDA', categorical_columns)
    if selected_categorical_columns:
        for column in selected_categorical_columns:
            fig, ax = plt.subplots()
            sns.countplot(x=data[column], ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

def run_models(data, numeric_columns, categorical_columns, target_variable, problem_type):
    X = data[numeric_columns].drop(columns=[target_variable]).fillna(0) if problem_type == 'Regression' else pd.get_dummies(data[categorical_columns + numeric_columns].drop(columns=[target_variable]), drop_first=True)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
    'Regression': [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge()),
        ('Lasso Regression', Lasso()),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor())
    ],
    'Classification': [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Decision Tree Classifier', DecisionTreeClassifier()),
        ('Random Forest Classifier', RandomForestClassifier()),
        ('SVM', SVC()),
        ('KNN', KNeighborsClassifier())
    ]
}


    selected_model = st.selectbox('Select Model', [model[0] for model in models[problem_type]])
    model = dict(models[problem_type])[selected_model]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if problem_type == 'Regression':
        mse = mean_squared_error(y_test, predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f'Model: {selected_model}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
    else:
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        st.write(f'Model: {selected_model}, Accuracy: {accuracy}')
        st.write('Confusion Matrix:', cm)

def run_h2o_automl(data, target_variable, problem_type):
    h2o_df = h2o.H2OFrame(data)
    x = [name for name in h2o_df.columns if name != target_variable]
    y = target_variable
    if problem_type == 'Classification':
        h2o_df[y] = h2o_df[y].asfactor()

    train, test = h2o_df.split_frame(ratios=[.8], seed=42)
    aml = H2OAutoML(max_models=15, max_runtime_secs=120, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    st.subheader('H2O AutoML Leaderboard')
    st.write(aml.leaderboard.as_data_frame())
    st.subheader('Best Model Performance')
    st.write(aml.leader.model_performance(test))

    if isinstance(aml.leader, h2o.estimators.H2OTreeModel):
        explainer = shap.TreeExplainer(aml.leader)
        shap_values = explainer.shap_values(h2o.as_list(test))
        st.pyplot(shap.summary_plot(shap_values, h2o.as_list(test), plot_type="bar"))

if __name__ == "__main__":
    main()
