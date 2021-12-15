
import streamlit as st
from io import StringIO
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def multivariateAnalysis_Correlation(df):
    #Plotting correlation plot
    st.header('Multivariate Analysis-Correlation plot')
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(df.corr(), cbar=True, annot=True, cmap='Blues')
    st.pyplot(fig)

def univariateAnalysis_Density(df):
    #Plotting Density plot
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
        
def univariateAnalysis_Features(df):
    #plotting features
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

def bivariateAnalysis_Activity_Gyroscope(df):
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
def bivariateAnalysis_Activity_Accelerometer(df):
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
if __name__ == '__main__':
    bivariateAnalysis_Activity_Accelerometer(pd.read_csv("../data/Data1.csv"))
