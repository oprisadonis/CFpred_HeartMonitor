from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def remove_nan(data_table, drop = False):
    nan_count = data_table.isna().sum()
    print(nan_count)
    if drop == True:
        data_table = data_table.dropna()
    else:
        if data_table.isnull().any().any() > 0:
            data_table = data_table.fillna(0)
    return data_table

def kNN_classifier(df):
    y = df
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    for number_of_neighbors in range (1,15):
        print(f"kNN classifier with number of neighbors: {number_of_neighbors}")
        col = "Sleepiness"
        data = y.loc[:, y.columns != col]
        target = y.loc[:, y.columns == col]

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)

        neigh = KNeighborsClassifier(n_neighbors=number_of_neighbors)  #kNN Regression algorithm with different neigbor numbers between 1-14
        neigh.fit(X_train, y_train.values.ravel())

        from sklearn.metrics import mean_squared_error

        inferred_body_mass = neigh.predict(X_test)
        model_error = mean_squared_error(y_test, inferred_body_mass)
        print(f"The mean squared error of the optimal model is {model_error:.2f}")

        from sklearn.metrics import mean_absolute_error

        model_error = mean_absolute_error(y_test, inferred_body_mass)
        print(f"The mean absolute error of the optimal model is {model_error:.2f} ")
        prediction = neigh.predict(X_test)
        recall = recall_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        print(f"recall: {recall:.2f}, precision: {precision:.2f}, accuracy: {accuracy:.2f}, f1-score: {f1:.2f}")
        print("confusion matrix:\n")
        print(confusion_matrix(y_test, prediction))
        print("\n\n")


def random_forest_classifier(df):
    y = df
    col = "Sleepiness"
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    for max_d in range(1,15):
        print(f"Random forest classifier with max depth: {max_d}")
        data = y.loc[:, y.columns != col]
        target = y.loc[:, y.columns == col]

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)
        clf = RandomForestClassifier(max_depth= max_d)
        clf.fit(X_train, y_train)

        from sklearn.metrics import mean_squared_error

        inferred_body_mass = clf.predict(X_test)
        model_error = mean_squared_error(y_test, inferred_body_mass)
        print(f"The mean squared error of the optimal model is {model_error:.2f}")

        from sklearn.metrics import mean_absolute_error

        model_error = mean_absolute_error(y_test, inferred_body_mass)
        print(f"The mean absolute error of the optimal model is {model_error:.2f} ")
        prediction = clf.predict(X_test)
        recall = recall_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        print(f"recall: {recall:.2f}, precision: {precision:.2f}, accuracy: {accuracy:.2f}, f1-score: {f1:.2f}")
        print("confusion matrix:\n")
        print(confusion_matrix(y_test, prediction))
        print("\n\n")

def svm_classifier(df):
    y = df
    col = "Sleepiness"
    from sklearn import svm
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    k = {"poly", "rbf", "sigmoid"} #"linear",
    for kernel_f in k:
        print(f"SVM classifier with kernel: {kernel_f}")
        data = y.loc[:, y.columns != col]
        target = y.loc[:, y.columns == col]

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)

        clf = svm.SVC(kernel = kernel_f)
        clf.fit(X_train, y_train.values.ravel())


        from sklearn.metrics import mean_squared_error

        inferred_body_mass = clf.predict(X_test)
        model_error = mean_squared_error(y_test, inferred_body_mass)
        print(f"The mean squared error of the optimal model is {model_error:.2f}")

        from sklearn.metrics import mean_absolute_error

        model_error = mean_absolute_error(y_test, inferred_body_mass)
        print(f"The mean absolute error of the optimal model is {model_error:.2f} ")
        prediction = clf.predict(X_test)
        recall = recall_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        print(f"recall: {recall:.2f}, precision: {precision:.2f}, accuracy: {accuracy:.2f}, f1-score: {f1:.2f}")
        print("confusion matrix:\n")
        print(confusion_matrix(y_test, prediction))
        print("\n\n")

def gaussian_NB_classifier(df):
    print("Gaussian Naive Bayes classifier\n")
    y = df
    col = "Sleepiness"
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    data = y.loc[:, y.columns != col]
    target = y.loc[:, y.columns == col]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import mean_squared_error

    model_error = mean_squared_error(y_test, y_pred)
    print(f"The mean squared error of the optimal model is {model_error:.2f}")

    from sklearn.metrics import mean_absolute_error

    model_error = mean_absolute_error(y_test, y_pred)
    print(f"The mean absolute error of the optimal model is {model_error:.2f} ")
    prediction = y_pred
    recall = recall_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    print(f"recall: {recall:.2f}, precision: {precision:.2f}, accuracy: {accuracy:.2f}, f1-score: {f1:.2f}")
    print("confusion matrix:\n")
    print(confusion_matrix(y_test, prediction))
    print("\n\n")

def multinomial_NB_classifier(df):
    print("Multinomial Naive Bayes classifier\n")
    y = df
    col = "Sleepiness"
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    data = y.loc[:, y.columns != col]
    target = y.loc[:, y.columns == col]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)
    clf = MultinomialNB()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import mean_squared_error

    model_error = mean_squared_error(y_test, y_pred)
    print(f"The mean squared error of the optimal model is {model_error:.2f}")

    from sklearn.metrics import mean_absolute_error

    model_error = mean_absolute_error(y_test, y_pred)
    print(f"The mean absolute error of the optimal model is {model_error:.2f} ")
    prediction = y_pred
    recall = recall_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    print(f"recall: {recall:.2f}, precision: {precision:.2f}, accuracy: {accuracy:.2f}, f1-score: {f1:.2f}")
    print("confusion matrix:\n")
    print(confusion_matrix(y_test, prediction))
    print("\n\n")

def mlp_classifier(df):
    print("MLP classifier")
    y = df
    col = "Sleepiness"
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
    data = y.loc[:, y.columns != col]
    target = y.loc[:, y.columns == col]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42, shuffle=False)

    estimator = MLPClassifier()

    param_grid = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 1)],
                  'activation': ['relu', 'tanh', 'logistic'],
                  'alpha': [0.0001, 0.05,0.5,0.007],
                  'learning_rate': ['constant', 'adaptive','invscaling'],
                  'solver': ['lbfgs', 'sgd','adam']}

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X_train, y_train)

    best_params = grid_result.best_params_

    best_mlp = MLPClassifier(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                            activation=best_params["activation"],
                            solver=best_params["solver"],
                            max_iter=5000, n_iter_no_change=200
                            )

    scoring = {
        'abs_error': 'neg_mean_absolute_error',
        'squared_error': 'neg_mean_squared_error',
        'r2': 'r2',
        'accuracy': 'accuracy'}

    scores = cross_validate(best_mlp, X_test, y_test, scoring=scoring, return_train_score=True, return_estimator=True)
    print(scores)
    print(f"\nBest parameters: {best_params}\n\n")

################################################## Main run functions

df = pd.read_csv("data_table\out_binary_sleepiness.csv")
print(df)
df = remove_nan(df,drop=True)
print(df)

####################################################
# nr_elem = df['Sleepiness'].count()
# nr_0 = df['Sleepiness'].value_counts()[0]
# nr_1 = df['Sleepiness'].value_counts()[1]
# print(f"\n\nNumber of elements: {nr_elem}\nNumber of zeros: {(nr_0/nr_elem)*100}%\nNumber of ones: {(nr_1/nr_elem)*100}\n\n\n")

df = df.sort_values(by=['lf','hf'],ascending=[False,False])
df = df.iloc[35:,:]
# features2 = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2',
#                 'breathingrate','vlf_perc', 'lf_perc', 'hf_perc']
# df.loc[:,features2]
# #
df = df.drop(df.query('Sleepiness == 0').sample(frac=.78).index)
#
# nr_elem = df['Sleepiness'].count()
# nr_0 = df['Sleepiness'].value_counts()[0]
# nr_1 = df['Sleepiness'].value_counts()[1]
# print(f"\n\nNumber of elements: {nr_elem}\nNumber of zeros: {(nr_0/nr_elem)*100}%\nNumber of ones: {(nr_1/nr_elem)*100}\n\n\n")

#######################################################

kNN_classifier(df)
gaussian_NB_classifier(df)
multinomial_NB_classifier(df)
svm_classifier(df)
random_forest_classifier(df)

mlp_classifier(df)

