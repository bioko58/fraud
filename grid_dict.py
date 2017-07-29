'''
DESCRIPTION:
     Defines, trains, and optimizes hyperparamters for a series of models.
     Returns the trained & tuned models, and their C.V. scores

USAGE:
    # import and call 'run_grid_search()' from within modeling file (e.g. model.py)
'''

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def grid_search_dict():
    '''
    defines all the models and hyperparameters to optimize with grid search.
    returns a list of tuples - (model, dict-of-hyper_params)
    '''

    gd_boost = {
        'learning_rate':[0.1, 0.05, 0.2, .25],
        'max_depth':[3, 2,4,8,10],
        'n_estimators':[25,180,500,800]}

    ada_boost = {
        'learning_rate':[0.1, 0.05, 0.2, 0.25],
        'max_depth':[3,2,8,10],
        'n_estimators':[25,180,500,800]}

    decision_tree = {
        'max_depth':[None,3,8],
        'min_samples_split':[2,10],
        'min_samples_leaf':[1,5]}

    random_forest_grid = {
        'n_estimators': [30,180,500],
        'min_samples_leaf': [2,6],
        }

    svm_grid = {
        'c': [10,50,250,500,900],
        'kernel': ['rbf', 'poly'],
        'gamma': ['auto']
        }


    # leave SVM out for now - takes too long and scores have been low
    return [
        (GradientBoostingClassifier(), gd_boost),
        (AdaBoostClassifier(), ada_boost),
        (RandomForestClassifier(), random_forest_grid),
        (DecisionTreeClassifier(), decision_tree)
    ]

#,

def run_grid_search(scoring_metric, X_train, y_train, printing=False):
    '''
    grid-searches a series of models and hyperparameters (as defined in grid_search_dict),
    to find the optimal hyperparameters for each model, that will optimized the provided scoring metric
    '''

    # creates models & their hyperparameter dicts
    models_and_hyperparams = grid_search_dict()


    # performs grid-search to tune hyperparameters for each model; stores results
    gs_results = []
    kf = KFold(n_splits=5, shuffle=False)

    for model, hyperparams in models_and_hyperparams:

        #gs = GridSearchCV(model, hyperparams, cv=kf, n_jobs=1 ,scoring=scoring_metric)
        gs = GridSearchCV(model, hyperparams, n_jobs=1, scoring=scoring_metric)
        gs.fit(X_train, y_train)
        gs_results.append(gs)


    # optionally prints the Grid search scores and hyperparam values (if printing parameter)
    if printing == True:
        for gs in gs_results:
            print "Grid Search results for: ", gs.best_estimator_.__class__.__name__
            print " params: ",gs.best_params_
            print "", gs.get_params()['scoring'], ": ", gs.best_score_
            print ""


    models = [gs.best_estimator_ for gs in gs_results]
    scores = [gs.best_score_ for gs in gs_results]
    # returns the models (tuned with their best hyper-parameters)
    return (models, scores)
