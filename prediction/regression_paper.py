from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix, make_scorer, \
    matthews_corrcoef, mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

import warnings

from sklearn.utils import resample

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  # or use an integer for a specific number
pd.set_option('display.width', 1000)

# flag for the train/test ratio
TESTSIZE = 0.2


def classifier_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'MCC': mcc}
    # print(f"Accuracy: {accuracy:.2f}   Precision: {precision:.2f}   Recall: {recall:.2f}   F1-score: {f1:.2f}\n")
    # print(f"Confusion_matrix:\n{confusion_matrix(y_true, y_pred)}\n")
    return pd.DataFrame(dict, index=[0])


def make_regression(df, model, labels_column, cv, number_of_runs):
    data = df.loc[:, df.columns != labels_column]
    target = df.loc[:, df.columns == labels_column]
    if cv:
        scorers = {
            'mean_squared_error': make_scorer(mean_squared_error),
            'r2': make_scorer(r2_score),
            'mean_absolute_error': make_scorer(mean_absolute_error),
            'explained_variance': make_scorer(explained_variance_score)
        }
        pipeline = Pipeline(steps=[
            ('oversampler', RandomOverSampler()),
            ('classifier', model)
        ])
        stratified_Kfold = StratifiedKFold(n_splits=5, shuffle=True)
        # for leave one out
        # kf = KFold(n_splits=len(df))
        scores = cross_validate(model, data, target, cv=stratified_Kfold, scoring=scorers, return_train_score=True)
        results = pd.DataFrame(scores)
        mean_results = results.mean()
        results.loc['Mean'] = mean_results.apply(lambda x: round(x, 3))
        if mean_results['test_mean_squared_error'] < 2:
            print("******** Training scores ********")
            print(results[['train_mean_squared_error', 'train_r2', 'train_mean_absolute_error',
                           'train_explained_variance']])
            print("******** Testing scores ********")
            print(
                results[['test_mean_squared_error', 'test_r2', 'test_mean_absolute_error', 'test_explained_variance']])


def kNN_regressor(df, labels_column, cv=False, number_of_runs=1):
    print("\n******************************* kNN classifier ***********************************************")
    for number_of_neighbors in range(3, 15, 2):
        print(f"*************** Number of neighbors: {number_of_neighbors}*************************\n")
        neigh = KNeighborsRegressor(n_neighbors=number_of_neighbors)
        make_regression(df, neigh, labels_column, cv, number_of_runs)


def linear_regression(df, labels_column, cv=False, number_of_runs=1):
    print("\n******************************* Linear Regression ***********************************************")
    linear_regressor = LinearRegression()
    make_regression(df, linear_regressor, labels_column, cv, number_of_runs)


def random_forest_regressor(df, labels_column, cv=False, number_of_runs=1):
    print("\n******************************* Random forest classifier ***********************************************")
    for n_trees in range(10, 100, 20):
        for max_d in range(1, 15, 3):
            print(f"********************* Max depth: {max_d} Number of threes: {n_trees}***********************\n")
            clf = RandomForestRegressor(max_depth=max_d, n_estimators=n_trees)
            make_regression(df, clf, labels_column, cv, number_of_runs)


def svm_regressor(df, labels_column, cv=False, number_of_runs=1):
    print("\n******************************* SVM classifier ***********************************************")
    k = {"poly", "rbf", "sigmoid", "linear"}
    for kernel_f in k:
        for c in range(1, 40, 5):
            print(f"***************** Kernel: {kernel_f} C: {c} ***********************\n")
            clf = svm.SVR(kernel=kernel_f, C=c)
            make_regression(df, clf, labels_column, cv, number_of_runs)

def mlp_regressor(df, labels_column, cv=False):
    print("\n******************** MLP classifier ******************************************")
    data = df.loc[:, df.columns != labels_column]
    target = df.loc[:, df.columns == labels_column]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    estimator = MLPRegressor()

    param_grid = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 1)],
                  'activation': ['relu', 'tanh', 'logistic'],
                  'alpha': [0.0001, 0.05, 0.5, 0.007],
                  'learning_rate': ['constant', 'adaptive', 'invscaling'],
                  'solver': ['lbfgs', 'sgd', 'adam']}

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X_train, y_train.values.ravel())

    best_params = grid_result.best_params_

    best_mlp = MLPClassifier(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                             activation=best_params["activation"],
                             solver=best_params["solver"],
                             max_iter=5000, n_iter_no_change=200
                             )

    scoring = {
        'mean_squared_error': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score),
        'mean_absolute_error': make_scorer(mean_absolute_error),
        'explained_variance': make_scorer(explained_variance_score)
    }

    scores = cross_validate(best_mlp, X_test, y_test, scoring=scoring, cv=5, return_train_score=True,
                            return_estimator=True)
    print(scores)
    print(f"\nBest parameters: {best_params}\n\n")


def run_regressors(df, cv=False, number_of_runs=1):
    kNN_regressor(df, "Sleepiness", cv, number_of_runs)
    random_forest_regressor(df, 'Sleepiness', cv, number_of_runs)
    linear_regression(df, 'Sleepiness', cv, number_of_runs)
    svm_regressor(df, 'Sleepiness', cv, number_of_runs)
    mlp_regressor(df, 'Sleepiness')


def identify_correlations_2(df):
    print(f"df shape: {df.shape}\n")
    y = df['Sleepiness']
    X = df.drop(columns=['Sleepiness'])
    X_binned = pd.DataFrame()

    for col in X.columns:
        X_binned[col] = pd.cut(X[col], bins=10, labels=False)

    # print(X_binned)

    # to use X_binned uncomment next line (chi2 does not use continues values)
    # X = X_binned

    # chi2
    # selector = SelectKBest(chi2, k='all')
    # ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')

    selector.fit(X, y)

    scores = selector.scores_
    p_values = selector.pvalues_

    results = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'p_value': p_values})
    results = results.sort_values('p_value', ascending=True)
    # print(results)

    X['Sleepiness'] = y
    return X, results


#df = pd.read_csv("data_table\out_non_binary_sleepiness.csv")
df = pd.read_csv("data_table\out_non_bs_normalized.csv")

df = df.sort_values(by=['lf', 'hf'], ascending=[False, False])
df = df.iloc[35:, :]
df.dropna(inplace=True)

print(df[df['Sleepiness'] == 1].shape)
print(df[df['Sleepiness'] == 2].shape)
print(df[df['Sleepiness'] == 3].shape)
print(df[df['Sleepiness'] == 4].shape)
print(df[df['Sleepiness'] == 5].shape)
print(df[df['Sleepiness'] == 6].shape)
print(df[df['Sleepiness'] == 7].shape)
x, rez = identify_correlations_2(df)
print(rez)

print(df.columns)
col1 = ['Sleepiness', 'bpm', 'lf/hf', 'ibi', 'hf_perc', 'vlf_perc', 'sdsd', 'sdnn', 'sd1/sd2',
        'breathingrate', 'sd1', 'sd2', 'rmssd', 'pnn50', 's', 'lf', 'pnn20', 'hr_mad',
        'p_total', 'vlf', 'hf', 'lf_perc']
col2 = ['Sleepiness', 'bpm', 'lf/hf', 'ibi', 'hf_perc', 'vlf_perc', 'sdsd', 'sdnn', 'sd1/sd2',
        'breathingrate']
df = df[col1]
run_regressors(df, cv=True)
