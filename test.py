import streamlit as st
from io import StringIO
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import logisticRegression
from models import decision_tree_with_KFold
from models import random_forest
from models import xgboost_classification
from models import neural_networks


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
            st.write(df)
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
            print(page)
            if page == "Multivariate Analysis-Correlation plot":
                st.header('Multivariate Analysis-Correlation plot')
                fig = plt.figure(figsize=(10, 4))
                sns.heatmap(df.corr(), cbar=True, annot=True, cmap='Blues')
                st.pyplot(fig)
            elif page == "Univariate Analysis-Density plot":
                st.header('Univariate Analysis-Density plot')
                plt.figure(figsize=(10, 15))
                facetgrid = sns.FacetGrid(df[df["subject"] == "subject1"],
                                          hue='Activity',
                                          height=6,
                                          aspect=2)
                facetgrid.map(sns.distplot, 'arx', hist=False).add_legend()
                st.pyplot(facetgrid)
                with st.expander("See Observations"):
                    st.write('For subject 1 and acceleration in the' +
                             ' direction of X Stationary Activities' +
                             ' and Dynamic Activities have a clear ' +
                             'distinguishment between them, Standing still,' +
                             ' sitting and relaxing, and lying down can be' +
                             ' classified with threshold values')
            elif page == "Univariate Analysis-Features":
                for col in ['grx', 'gly', 'aly', 'alz']:
                    facetgrid = plt.figure(figsize=(10, 10))
                    plt.title('Feature: {}'.format(col))
                    plt.xticks(rotation=40)
                    sns.boxplot(x='Activity',
                                y=col,
                                data=df,
                                showfliers=False,
                                saturation=1)
                    st.pyplot(facetgrid)
            elif page == "Bivariate Analysis-Activity VS Gyroscope Readings":
                st.header("Bivariate Analysis-Activity VS Gyroscope Readings")
                for activity in ["Jogging", "Lying down"]:
                    fig4 = plt.figure(figsize=(16, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['glx'],
                             alpha=.7, label="{}: glx".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['gly'],
                             alpha=.7, label="{}: gly".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['glz'],
                             alpha=.7, label="{}: glz".format(activity))
                    plt.title('{} - left-hand'.format(activity))
                    plt.legend(loc='upper right')
                    plt.subplot(1, 2, 2)
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['grx'],
                             alpha=.7, label="{}: grx".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['gry'],
                             alpha=.7, label="{}: gry".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['grz'],
                             alpha=.7, label="{}: grz".format(activity))
                    plt.title('{} - right-hand'.format(activity))
                    plt.legend(loc='upper right')
                    st.pyplot(fig4)
            elif page == "Bivariate Analysis-Activity VS Accelerometer Readings":
                st.header("Bivariate Analysis-Activity VS" +
                          " Accelerometer Readings")
                for activity in ["Standing still", "Running"]:
                    fig6 = plt.figure(figsize=(16, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['alx'],
                             alpha=.7, label="{}: alx".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['aly'],
                             alpha=.7, label="{}: aly".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['alz'],
                             alpha=.7, label="{}: alz".format(activity))
                    plt.title('{} - left-hand'.format(activity))
                    plt.legend(loc='upper right')
                    plt.subplot(1, 2, 2)
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['arx'],
                             alpha=.7, label="{}: arx".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['ary'],
                             alpha=.7, label="{}: ary".format(activity))
                    plt.plot(df[df['Activity'] == activity].reset_index(drop=True)['arz'],
                             alpha=.7, label="{}: arz".format(activity))
                    plt.title('{} - right-hand'.format(activity))
                    plt.legend(loc='upper right')
                    st.pyplot(fig6)
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
