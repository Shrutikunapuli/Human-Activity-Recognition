from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from load_datasets import load_dataset
import mlflow


def logisticRegression():
    mlflow.sklearn.autolog()
    experiment_id = mlflow.set_experiment("Logistic Regression Model")
    with mlflow.start_run(run_name='logistic regression') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        lr = LogisticRegression(penalty="l2", solver='liblinear')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print(y_pred)
        metrics = mlflow.sklearn.eval_and_log_metrics(lr,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def decision_tree():
    mlflow.sklearn.autolog()
    experiment_id = mlflow.set_experiment("Decision Trees")
    with mlflow.start_run(run_name='decision tree') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        parameters = {'min_samples_split': range(2, 5, 10),
                      'max_depth': range(2, 5, 10)}
        clf_tree = DecisionTreeClassifier()
        clf = GridSearchCV(clf_tree, parameters)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(clf,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def decision_tree_with_KFold(n_splits=10):
    experiment_id = mlflow.set_experiment("Decision Trees with" +
                                          "K-Fold Cross Validation")
    with mlflow.start_run(run_name='decision tree') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        clf_tree = DecisionTreeClassifier()
        k_fold = KFold(n_splits)
        mlflow.log_param("n_splits", n_splits)
        accuracy = cross_val_score(clf_tree, X_train, y_train, cv=k_fold)
        y_pred = cross_val_predict(clf_tree, X_test, y_test)
        mlflow.log_metric("training_score", accuracy.mean())
        accuracy = accuracy_score(y_pred, y_test)
        mlflow.log_metric("test_score", accuracy)
        mlflow.end_run()


def random_forest():
    experiment_id = mlflow.set_experiment("Random Forest")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name='random forest') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(classifier,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def xgboost_classification():
    experiment_id = mlflow.set_experiment("XGBoost")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name='xgboost') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(classifier,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def neural_networks():
    experiment_id = mlflow.set_experiment("Neural Networks")
    mlflow.keras.autolog()
    with mlflow.start_run(run_name='neural networks') as run:
        X_train, X_test, y_train, y_test = load_dataset('../data/mhealth_raw_data 2.csv')
        model = Sequential()
        model.add(Dense(units=64,
                        kernel_initializer='normal',
                        activation='sigmoid',
                        input_dim=X_train.shape[1]))
        model.add(Dropout(0.2))
        model.add(Dense(units=13,
                  kernel_initializer='normal',
                  activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        mo_fitt = model.fit(X_train, y_train, epochs=75,
                            validation_data=(X_test, y_test))
        mlflow.end_run()

if __name__ == '__main__':
    neural_networks()
