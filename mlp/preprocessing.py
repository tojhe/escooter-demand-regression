'''
Script to load, clean, transform and split data
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re, argparse

def cli_args():
    '''
    CLI arguments
    :return: parse_args obj - args containing the input argument values
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '-testsize', dest='testsize', type=float,
                        help="Test size as decimal proportion of dataset", required=True)
    parser.add_argument('-l', '-load', dest='load', type=str,
                        help="URL for dataset", required=True)
    args = parser.parse_args()
    return args

def read_data(file_path):
    '''
    Reads data from source
    :param file_path: str - file/URL path of data
    :return: pd.DataFrame of full dataset
    '''
    df = pd.read_csv(file_path, parse_dates=['date'])
    # Replacing '-' in column names with '_'
    df.columns = [col.replace('-','_') for col in df.columns]
    return df

def datetime_processing(df):
    '''
    Retrieve features from date
    :param df: pd.DataFrame - full dataset
    :return: -
    '''
    df['datetime'] = df['date'] + pd.to_timedelta(df['hr'], unit='h')
    df['day_in_week'] = df['date'].apply(lambda x: x.day_name().lower()[:3])
    df['month'] = df['date'].apply(lambda x: x.month_name().lower()[:3])

def missing_obsv(df):
    '''
    Handles missing observations in the dataset.
    Removes dates with less than 22 original readings in a day.
    Forward fills originally missing hourly data in a day.
    :param df: pandas Dataframe of original data
    :return: missing reading handled dataset Dataframe
    '''

    df_dr = pd.DataFrame(pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='H'),
                         columns=['datetime'])
    df = pd.merge(df, df_dr, how='right', on='datetime')
    df['date'] = df['datetime'].apply(lambda x: x.date())

    # Select dates with less than 22 original hourly observation to remove
    dates_to_remove = df.groupby('date')['hr'].count()[df.groupby('date')['hr'].count() < 22].index
    # Drop the dates
    df.drop(axis=0, labels=df[df['date'].isin(dates_to_remove)].index, inplace=True)

    return df

def error_cleaning(df):
    '''
    Handles wrong user volume data and missing observations
    :param df: pd.DataFrame - original dataset
    :return: pd.DataFrame - cleaned dataset
    '''
    # print (df.shape)
    df.drop_duplicates(inplace=True, keep='first')

    # handle missing readings
    df = missing_obsv(df)

    # convert negative users volume as null
    df['guest_users'] = df['guest_users'].apply(lambda x: np.NaN if x < 0 else x)
    df['registered_users'] = df['registered_users'].apply(lambda x: np.NaN if x < 0 else x)

    #forward fill null values
    df.ffill(axis=0, inplace=True,)

    return df

def clean_weather(weather_string):
    '''
    Corrects erroneous weather values
    :param weather_string: str
    :return: str -  weather_string
    '''
    weather_string = weather_string.lower()
    if weather_string == 'lear':
        return 'clear'
    elif weather_string == 'loudy':
        return 'cloudy'
    elif re.match(r'.*[ //].*', weather_string):
        weather_string = weather_string.replace(' ', '_')
        weather_string = weather_string.replace('/','_')
        return weather_string

    return weather_string

def feat_create(df):
    '''
    Creates new features
    :param df: pd.DataFrame - Original data
    :return: pd.DataFrame - Newly feature engineering data
    '''

    df['hr'] = df['hr'].astype('int').astype('O')

    # Dummy coding and merge to original data
    df = df.merge(pd.get_dummies(df[['hr', 'weather', 'day_in_week', 'month']], prefix=['hr', 'weather', 'daywk', 'mth'], drop_first=False), left_index=True, right_index=True)

    # Secondary target variables - first differencing
    df['registered_1diff'] = df['registered_users'] - df['registered_users'].shift(24)
    df['guest_1diff'] = df['guest_users'] - df['guest_users'].shift(24)

    # Main target variable
    df['total_users'] = df['guest_users'] + df['registered_users']

    return df


if __name__ == '__main__':

    # parse cli args
    args = cli_args()

    # load data
    df = read_data(args.load)

    # processing
    datetime_processing(df)
    df = error_cleaning(df)
    df['weather'] = df['weather'].apply(clean_weather)
    df = feat_create(df)

    # date chronology sort
    df.sort_values("datetime", inplace=True, ascending=True)
    # dropping columns not needed anymore
    df.drop(axis=1, labels=['weather', 'day_in_week', 'month', 'hr', 'date', 'datetime'], inplace=True)

    # drop rows with null values, especially targeting null in 1diff features --> the first and last 24 hours of the dataset
    df.dropna(axis=0,how='any',inplace=True)


    # Split data
        # Select guest/registered/total in column name as features as target variables
    df_target = df.filter(regex=r'^guest|^registered|^total')
        # Select the rest of remaining features as predictors
    df_predictor = df[[col for col in df.columns if col not in df_target.columns]]
        # Split data
    X_train, X_test, y_train, y_test = train_test_split(df_predictor,
                                                        df_target,
                                                        test_size=args.testsize, shuffle=False)
    # Min Max Scaling, fitting against training data
    with pd.option_context('mode.chained_assignment', None):
        numeric_features = ['temperature', 'feels_like_temperature', 'relative_humidity', 'windspeed', 'psi']
        ss = MinMaxScaler()
        X_train_s = ss.fit_transform(X_train[numeric_features])
        X_train.loc[:, numeric_features] = X_train_s
        X_test_s = ss.transform(X_test[numeric_features])
        X_test.loc[:, numeric_features] = X_test_s

    # Reset indexes
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # Export
    X_train.to_pickle('./mlp/data/X_train.pkl')
    X_test.to_pickle('./mlp/data/X_test.pkl')
    y_train.to_pickle('./mlp/data/y_train.pkl')
    y_test.to_pickle("./mlp/data/y_test.pkl")
