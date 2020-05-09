'''
Script containing all the modeling functions
'''
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from math import sqrt

def read_files():
    '''
    Reads pickled training and test files
    :return: pd.DataFrame - X_train, X_test, y_train, y_test
    '''
    X_train = pd.read_pickle('./mlp/data/X_train.pkl')
    y_train = pd.read_pickle('./mlp/data/y_train.pkl')
    X_test = pd.read_pickle('./mlp/data/X_test.pkl')
    y_test = pd.read_pickle('./mlp/data/y_test.pkl')

    return X_train, X_test, y_train, y_test

def reverse_difference(y_predict, user_category, y_test, y_train):
    '''
    Adds predicted difference to actual user volume from 24 observations ago
    :param y_predict: np.array - predicted target
    :param user_category:str - user group to convert, registered or guest
    :param y_test: pd.DataFrame - target test dataset
    :param y_train: pd.DataFrame - target train dataset
    :return:
    '''
    volume = np.array([])
    # convert first 24 hours of predicted, with actual user values only available in last 24 observations of training set
    volume = np.append(volume, y_train[user_category].iloc[-24:] + y_predict[:24])
    # similar to previous command, but on test set actual past 24 observations user value
    for i, val in enumerate(y_predict[24:]):
        volume = np.append(volume, y_test[user_category].values[i] + val)

    return volume

def aggregate_users(y_predict_reg, y_predict_gue):
    '''
    Generates predicted total user values
    :param y_predict_reg: predicted registered users
    :param y_predict_gue: predicted guest users
    :return: predicted total user
    '''
    return y_predict_reg + y_predict_gue

def run_model(X_train, y_train, X_test, estimator_name='lasso', param_grid=None, n_splits=5, target_group='registered'):
    '''
    Fits a time series cross-validated model of choice.
    :param X_train: pd.DataFrame Train predictor variables
    :param y_train: pd.DataFrame - Train target variables
    :param X_test: pd.DataFrame - Test predictor variables
    :param estimator_name: str - estimator to use (lasso, randomforest, adaboost)
    :param param_grid: dict - paramaters grid for gridsearch
    :param n_splits: int - number of splits for cross-validation
    :param user_group: str - registered or guests users
    :return: y_predict, model
    '''
    if target_group == 'registered':
        target = 'registered_1diff'
    elif target_group == 'guest':
        target = 'guest_1diff'
    else:
        raise Exception("Invalid user group keyed! \n Enter 'registered' or 'guest")


    tscv = TimeSeriesSplit(n_splits=n_splits)

    if estimator_name=='lasso':
        print ('Training Lasso model')
        model = LassoCV(n_jobs=-1, cv=tscv, random_state=42)
        model.fit(X_train, y_train[target])


    else:
        if estimator_name=='randomforest':
            print('Training Random Forest model')
            est = RandomForestRegressor(random_state=42)

        elif estimator_name=='gradientboost':
            print ('Training Gradient Boost model')
            est = GradientBoostingRegressor(random_state=42)

        elif estimator_name=='xgboost':
            print ('Training xgboost model')
            est = XGBRegressor(random_state=42)

        model = GridSearchCV(estimator=est, param_grid=param_grid, cv=tscv, n_jobs=-1)

    model.fit(X_train, y_train[target])
    print ("Training completed")
    y_predict = model.predict(X_test)

    return y_predict, model

def evaluate(y_test, y_predict, y_train=None, estimator_name='lasso',aggregate=False, target_group='registered'):
    '''
    Evaluates the model performance
    :param y_test: pd.DataFrame -  target test data
    :param y_predict: np.array - predicted target
    :param y_train: pd.DataFrame - target train data
    :param estimator_name: str - name of estimator used
    :param aggregate: bool - whether to run addition from first difference (True is to run)
    :param target_group: str - registered/guest/total
    :return:
    '''
    if aggregate:
        if target_group == 'registered':
            target = 'registered_users'
        elif target_group == 'guest':
            target = 'guest_users'
        else:
            raise Exception("Invalid user group keyed! \n Enter 'registered' or 'guest")
        y_predict_agg = reverse_difference(y_predict, target, y_test, y_train)
        rmse = sqrt(mean_squared_error(y_test[target], y_predict_agg))
        r2= r2_score(y_test[target], y_predict_agg)

    else:
        if target_group == 'total':
            target = 'total_users'
        else:
            raise Exception("Invalid user group keyed! \n Enter 'total' or set aggregate to True")

        rmse = sqrt(mean_squared_error(y_test[target], y_predict))
        r2 = r2_score(y_test[target], y_predict)

    print ("{} performance on {} user volume".format(estimator_name, target_group))
    print ("rmse: ", rmse)
    print ("r2: ", r2)

    if target_group=='total':
        return rmse, r2
    if aggregate:
        return y_predict_agg, rmse, r2
