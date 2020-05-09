'''
Script to execute model training and evaluations
'''
from modeling import *
import pandas as pd
import argparse, configparser, ast, pickle

def extract_param_grid(model_params, file_path='./mlp/data/models_config.yml'):
    '''
    Extract parameter grid from configuration file
    :param model_params: str - model param identifier in config file
    :param file_path: str - file path of config file
    :return: dict - param_grid
    '''
    config = configparser.RawConfigParser()
    config.read(file_path)
    param_grid = {}
    for option in config.options(model_params):
        param_grid[option] = ast.literal_eval(config.get(model_params, option))

    return param_grid


def cli_args():
    '''
    CLI arguments
    :return: parse_args obj - args containing the input argument values
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--all', dest='all', type=str,
                        help="Run full evaluation. Y - Train all model, N - Train selected model", required=True)
    parser.add_argument('-r', '--registered', dest='reg', type=int,
                        help="Model to train for registered users. Select number: \n1 -Lasso\n|2 - Random Forest\n|3 - Gradient Boosting\n|4 - xgboost",
                        required=False)
    parser.add_argument('-g', '--guest', dest='gue', type=int,
                        help="Model to train for guest users. Select number:\n1 - Lasso\n|2 - Random Forest\n|3 - Gradient Boosting\n|4 - xgboost",
                        required=False)

    args = parser.parse_args()

    return args

def test_eval_model(X_train, y_train, X_test, y_test, estimator_name, param_grid=None, on='all'):
    '''
    Run evaluations
    :param X_train: pd.DataFrame - predictor train data
    :param y_train: pd.DataFrame - target train data
    :param X_test: pd.DataFrame - predictor test data
    :param y_test: pd.DataFrame - target test data
    :param estimator_name: str - name of estimator used
    :param param_grid: dict - parameter grid for grid search
    :param on: str - scope of user group test 'all'/'registered'/'guest'
    :return:
    '''
    if on == 'all':
        # On all user groups
        # Registered users
        y_predict_r, model_r = run_model(X_train, y_train, X_test, estimator_name,param_grid=param_grid)
        predicted_reg, rmse_r, r2_r = evaluate(y_test, y_predict_r, y_train, estimator_name, aggregate=True)

        # Guest users
        y_predict_g, model_g = run_model(X_train, y_train, X_test, estimator_name, param_grid=param_grid, target_group='guest')
        predicted_gue, rmse_g, r2_g = evaluate(y_test, y_predict_g, y_train, estimator_name, aggregate=True, target_group='guest')
        print (rmse_g, "guest")

        return predicted_reg, predicted_gue, model_r, model_g, rmse_r, r2_r, rmse_g, r2_g

    elif on == 'registered':
        # On registered only
        y_predict_r, model_r = run_model(X_train, y_train, X_test, estimator_name,param_grid=param_grid)
        predicted_reg, rmse_r, r2_r = evaluate(y_test, y_predict_r, y_train, estimator_name, aggregate=True)

        return predicted_reg, model_r, rmse_r, r2_r

    elif on == 'guest':
        # On guest only
        y_predict_g, model_g = run_model(X_train, y_train, X_test, estimator_name, param_grid=param_grid, target_group='guest')
        predicted_gue, rmse_g, r2_g = evaluate(y_test, y_predict_g, y_train, estimator_name, aggregate=True, target_group='guest')

        return predicted_gue, model_g, rmse_g, r2_g




if __name__=='__main__':
    # parser
    args = cli_args()
    #read files
    X_train, X_test, y_train, y_test = read_files()

    # If user selects to run full evaluations
    if args.all.lower() == 'y':
        models = [{'estimator_name':'lasso', 'param_grid':extract_param_grid('lasso_param_grid')},
                  {'estimator_name': 'randomforest', 'param_grid': extract_param_grid('randomf_param_grid')},
                  {'estimator_name': 'gradientboost', 'param_grid': extract_param_grid('gradientb_param_grid')},
                  {'estimator_name': 'xgboost', 'param_grid': extract_param_grid('xgb_param_grid')}
                  ]

        #best scores/model init _r for registered, _g for guest
        best_score_r = 0
        best_score_g = 0
        best_name_r = ''
        best_name_g = ''
        best_model_g = None
        best_model_r = None
        best_prediction_r = []
        best_prediction_g = []

        #metrics dict for dataframe and csv export later
        metrics = {'model': [], 'registered_users_r2': [], 'registered_users_rmse':[], 'guest_users_r2': [], 'guest_users_rmse': []}

        # Test for all models on both registered and guest
        for mod in models:
            predicted_reg, predicted_gue, model_r, model_g, rmse_r, r2_r, rmse_g, r2_g = test_eval_model(X_train, y_train, X_test,y_test, mod['estimator_name'],param_grid=mod['param_grid'])
            metrics['model'].append(mod['estimator_name'])
            metrics['registered_users_rmse'].append(rmse_r)
            metrics['registered_users_r2'].append(r2_r)
            metrics['guest_users_rmse'].append(rmse_g)
            metrics['guest_users_r2'].append(r2_g)

            if r2_r > best_score_r:
                best_score_r = r2_r
                best_name_r = mod['estimator_name']
                best_model_r = model_r
                best_prediction_r = predicted_reg
            if r2_g > best_score_g:
                best_score_g = r2_g
                best_name_g = mod['estimator_name']
                best_model_g = model_g
                best_prediction_g = predicted_gue
            print ('------///------------///------')

        print ("The best model for registered users is {}.\n The best model for guest users is {}.".format(best_name_r, best_name_g))

        # Aggregate performance
        predicted_tot = aggregate_users(best_prediction_r, best_prediction_g)
        rmse_t, r2_t = evaluate(y_test, predicted_tot, y_train, 'combined model', aggregate=False, target_group='total')

        # Overall Metrics
        metrics_total_df = pd.DataFrame.from_dict({'total_users_rmse': [rmse_t], 'total_users_r2': [r2_t] })
        metrics_df = pd.DataFrame.from_dict(metrics)

        print (metrics_df)
        print (metrics_total_df)

        # Outputs
        metrics_total_df.to_csv('./mlp/output/metrics_total.csv', index=False)
        metrics_df.to_csv('./mlp/output/metrics_reg_gue.csv', index=False)

        with open('./mlp/output/reg_model_{}.pkl'.format(best_name_r), 'wb') as r_file:
            pickle.dump(best_model_r, r_file)
        with open('./mlp/output/gue_model_{}.pkl'.format(best_name_g), 'wb') as g_file:
            pickle.dump(best_model_g, g_file)
        with open('./mlp/output/total_predict.pkl', 'wb') as file:
            pickle.dump(predicted_tot, file)


    elif args.all.lower() == 'n':
        # selects not to run all -a n
        models = {1: {'estimator_name': 'lasso', 'param_grid': extract_param_grid('lasso_param_grid')},
                  2: {'estimator_name': 'randomforest', 'param_grid': extract_param_grid('randomf_param_grid')},
                  3: {'estimator_name': 'gradientboost', 'param_grid': extract_param_grid('gradientb_param_grid')},
                  4: {'estimator_name': 'xgboost', 'param_grid': extract_param_grid('xgb_param_grid')}
                 }
        # if selects model for registered -r int
        if args.reg:
            print('for registered users\n model selected: ', models[args.reg]['estimator_name'])
            print('param_grid: ', models[args.reg]['param_grid'])
            predicted_reg, model_r, rmse_r, r2_r = test_eval_model(X_train, y_train, X_test, y_test,
                                                                   models[args.reg]['estimator_name'],
                                                                   param_grid=models[args.reg]['param_grid'], on='registered')
        # if selects model for guest -g int
        if args.gue:
            print('for guest users\n model selected: ', models[args.gue]['estimator_name'])
            print('param_grid: ', models[args.gue]['param_grid'])
            predicted_reg, model_r, rmse_r, r2_r = test_eval_model(X_train, y_train, X_test, y_test,
                                                                   models[args.gue]['estimator_name'],
                                                                   param_grid=models[args.gue]['param_grid'], on='guest')

    else:
        raise Exception("Please use y or n for cli argument for -a")