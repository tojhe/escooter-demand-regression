'''
Generate parameter grid configurations for model Grid Search
'''

import configparser

config = configparser.ConfigParser()

config['lasso_param_grid'] = {'alpha': [0.1, 0.7, 1.0, 1.2]}
config['randomf_param_grid'] = {'n_estimators':[100, 200, 300, 500],
                                'min_samples_split': [2, 3, 4, 10, 15],
                                'max_features': ['auto', 'sqrt']}
config['gradientb_param_grid'] = {'n_estimators': [100, 200, 300, 400, 600],
                                  'learning_rate': [0.01, 0.1, 1.0],
                                  'loss': ['ls', 'lad', 'huber']}
config['xgb_param_grid'] = {'nthread':[1,2],
                            'objective':['reg:linear'],
                            'learning_rate': [0.1, 0.01, 0.001],
                            'max_depth': [5],
                            'min_child_weight': [4],
                            'silent': [1],
                            'subsample': [0.7, 0.8],
                            'n_estimators': [200, 300, 400, 500, 600]}

with open('./mlp/data/models_config.yml', 'w') as configfile:
    config.write(configfile)
