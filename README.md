# HUMAN ACTIVITY RECOGNITION

## Motivation

By monitoring humans’ daily actions, we can gain insight into their psychological condition and personality. Following this pattern, researchers aspire to use algorithms to anticipate human nature, and Human Activity Recognition is a popular field of research. Activity Recognition (AR) is monitoring the liveliness of a person by using intelligent devices such as smart phones and wearable devices. These devices capture sensor and motion data to classify human activities.


## Method and results

We used a dataset from the UCI machine learning repository to determine the accuracy of five machine learning models and perform statistical significance tests in this research. Human behaviors are divided into eight categories in the dataset: Sitting, Walking, Standing, Laying, Walking Upstairs, Walking Downstairs, Running, and Jogging.A variety of machine learning and deep learning techniques, such as Decision Tree XGBoost, Random Forest, and Artificial Neural Network are tested on this data. The problem of classifying daily human activity using data collected from smartphone sensors is referred to as human activity recognition.

Random forest has the best accuracy- 95% on test data when implemented with Grid Search.

For UI and Data Visualization: Streamlit
It is an open-source app to create apps.

``` bash
pip install streamlit #Install streamlit

streamlit hello #demo
```
Streamlit can also be installed in a virtual environment on Windows, Mac, and Linux.

For Tracking and Logging Model Parameters: MLFlow
Simplifies end-to-end machine learning lifecycle.
 


## Repository overview
```bash
├── README.md         
├── requirements.txt   #Contains all the extra libraries
├── SML.pdf            # brief intro PPT to the approach
├── data
│   ├── Data1.csv      #Data to visualize on streamlit     
│   └── mhealth_raw_data 2.csv (Download from Gdrive) #entire data for training
└── src
    ├── models.py   #Contains all models
    ├── EDA.py      # Visualization code
    ├── load_datasets.py  # Loading dataset
    └── train.py    # training and UI 
 ```

## Running instructions

1) Download data from  [link](https://drive.google.com/file/d/14RkZYl9BdzFaOpZimL9FPRpIrWGEsbMY/view?usp=sharing)
Unzip and place the mhealth_raw_data 2.csv in data folder

2) Run ``` pip install -r requirements.txt ```
Install all the required libraries 

3) Go to src and Run ``` streamlit run train.py ```

Another browser will open like below(http://localhost:8502/)-

![image](https://user-images.githubusercontent.com/29593466/146152018-9d2254e3-00d3-49a0-bf75-90f46677ae1e.png)

Upload data/Data1.csv file through browse button
- Understand the data 
- Select the desired plot
- For model building select desired option on dropdown

![image](https://user-images.githubusercontent.com/29593466/146152041-5e1104c2-22fe-4c0f-b8ad-3a7eaad9c5b8.png)


4) To view models on MLFlow UI click on http link after modelling (In gernal: http://127.0.0.1:5000/)
and 
Go to /src and run ``` mlflow ui ```
and refresh

![image](https://user-images.githubusercontent.com/29593466/146152075-947ad203-5245-4af8-bc20-e62504c58114.png)
 
Few test cases
 ``` assert(len(df)!=0),"file not selected" 
assert  pd.notnull(df).all().all(),"The has nulls"
assert df.shape[1] == 14, "Expected 14 columns" 
``` 
Output ![image](https://user-images.githubusercontent.com/29593466/146166968-0552f620-38da-49ec-b7ba-8f64a6feaaff.png)


## More resources

Mlflow docs - https://mlflow.org/docs/latest/index.html

Streamlit docs - https://docs.streamlit.io/

## Results 
Metrics for each model can be viewd on MLflow. 

Highest accuracy can be observed for Random forest algorith ~95%
