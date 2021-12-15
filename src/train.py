import streamlit as st
from io import StringIO
import pandas as pd
from models import logisticRegression
from models import decision_tree_with_KFold
from models import random_forest
from models import xgboost_classification
from models import neural_networks
from EDA import multivariateAnalysis_Correlation
from EDA import univariateAnalysis_Density
from EDA import univariateAnalysis_Features
from EDA import bivariateAnalysis_Activity_Gyroscope
from EDA import bivariateAnalysis_Activity_Accelerometer
import mlflow

def main():
    page = st.sidebar.selectbox(
        "Select a plot:",
        ["select",
            "Multivariate Analysis-Correlation plot",
            "Univariate Analysis-Density plot",
            "Univariate Analysis-Features",
            "Bivariate Analysis-Activity VS Gyroscope Readings",
            "Bivariate Analysis-Activity VS Accelerometer Readings"]
    )
    st.title('HUMAN ACTIVITY RECOGNITION')
    uploaded_file = st.file_uploader("Choose data file")
    if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            assert  pd.notnull(df).all().all(),"The has nulls" # check if the df has nulls
            st.write(df)
            assert df.shape[1] == 14, "Expected 14 columns" # else activity may cause error
            #assert len(set(df["Activity"].unique())) == 13, "Expected 13 activities"
            df['Activity'] = df['Activity'].replace([1, 2, 3, 4, 5, 6,
                                                     7, 8, 9, 10, 11, 12],
                                                    ['Standing still',
                                                     'Sitting and relaxing',
                                                     'Lying down',
                                                     'Walking',
                                                     'Climbing stairs',
                                                     'Waist bends forward',
                                                     'Front elevation of arms',
                                                     'Knees bending',
                                                     'Cycling',
                                                     'Jogging',
                                                     'Running',
                                                     'Jump front & back'])
            #print(page)
            if page == "Multivariate Analysis-Correlation plot":
                multivariateAnalysis_Correlation(df)
            elif page == "Univariate Analysis-Density plot":
                univariateAnalysis_Density(df)
            elif page == "Univariate Analysis-Features":
                univariateAnalysis_Features(df)
            elif page == "Bivariate Analysis-Activity VS Gyroscope Readings":
                bivariateAnalysis_Activity_Gyroscope(df)
            elif page == "Bivariate Analysis-Activity VS Accelerometer Readings":
                bivariateAnalysis_Activity_Accelerometer(df)
            
    model = st.selectbox("Select a model:",
                         ("select", 'Logistic Regression',
                          'Decision Tree with KFold',
                          'Random Forest',
                          'XGboost',
                          'Neural Networks'))
    if model == "select":
        mlflow.end_run()
    elif model == 'Logistic Regression':
        st.write('Training Logistic Regression. PLEASE WAIT......')
        logisticRegression()
        st.write('Done with model building please go to' +
                 '-http://127.0.0.1:5000/')
    elif model == 'Decision Tree with KFold':
        st.write("Training Decision Tree. PLEASE WAIT......")
        decision_tree_with_KFold(n_splits=10)
        st.write('Done with model building please go to' +
                 '-http://127.0.0.1:5000/')
    elif model == 'Random Forest':
        st.write("Training Random Forest. PLEASE WAIT......")
        random_forest()
        st.write('Done with model building please go to' +
                 '-http://127.0.0.1:5000/')

    elif model == 'XGboost':
        st.write("Training XGboost.  PLEASE WAIT......")
        xgboost_classification()
        st.write('Done with model building please go to' +
                 '-http://127.0.0.1:5000/')

    else:
        st.write("Training Neural Networks. PLEASE WAIT......")
        neural_networks()
        st.write('Done with model building please go to' +
                 '-http://127.0.0.1:5000/')


if __name__ == "__main__":
    main()
