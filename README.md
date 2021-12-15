# HUMAN ACTIVITY RECOGNITION

## Motivation

By monitoring humans’ daily actions, we can gain insight into their psychological condition and personality. Following this pattern, researchers aspire to use algorithms to anticipate human nature, and Human Activity Recognition is a popular field of research. Activity Recognition (AR) is monitoring the liveliness of a person by using intelligent devices such as smart phones and wearable devices. These devices capture sensor and motion data to classify human activities.


## Method and results

We used a dataset from the UCI machine learning repository to determine the accuracy of five machine learning models and perform statistical significance tests in this research. Human behaviors are divided into eight categories in the dataset: Sitting, Walking, Standing, Laying, Walking Upstairs, Walking Downstairs, Running, and Jogging.A variety of machine learning and deep learning techniques, such as Decision Tree XGBoost, Random Forest, and Artificial Neural Network are tested on this data. The problem of classifying daily human activity using data collected from smartphone sensors is referred to as human activity recognition.

Random forest has the best accuracy- 95% on test data when implemented with Grid Search.

For UI and Data Visualization: Streamlit
It is an open-source app to create apps.

pip install streamlit

streamlit hello

Streamlit can also be installed in a virtual environment on Windows, Mac, and Linux.



## Repository overview
```bash
├── README.md
├── requirements.txt
├── data
│   ├── Data1.csv
│   └── mhealth_raw_data 2.csv (Download from Gdrive)
└── src
    ├── models.py
    ├── EDA.py
    ├── load_datasets.py
    └── train.py
 ```

## Running instructions



## More resources

Point interested users to any related literature and/or documentation.


## About

Explain who has contributed to the repository. You can say it has been part of a class you've taken at Tilburg University.


Download data from  [link](https://drive.google.com/file/d/14RkZYl9BdzFaOpZimL9FPRpIrWGEsbMY/view?usp=sharing)


pip install -r requirements.txt

### Installation
pip install streamlit

streamlit hello

Streamlit can also be installed in a virtual environment on Windows, Mac, and Linux.
