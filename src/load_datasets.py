import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path, test_size=0.2, random_state=12):
    data = pd.read_csv(path)
    data = data[data['Activity'] != 0]
    data_req = data.groupby(by=['Activity']).sample(5000,
                                                    random_state=random_state)
    Y = data_req['Activity']
    X = data_req.drop(['Activity', "subject"], axis=1)
    X = pd.DataFrame(StandardScaler().fit_transform(X))
    mlflow.log_param("dataset_path", path)
    mlflow.log_param("dataset_shape", data_req.shape)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    return train_test_split(X, Y,
                            test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset(
        '../data/Data1.csv', ';'
    )
    print(x_train.head())
    print(y_train.head())
