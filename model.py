'''
DESCRIPTION:
    - Loads the featurized data,
    - Trains and tunes/optimizes numerous candidate models,
    - Scores each candidate model,
    - Selects highest scoring model,
    - Saves top model into './model.pkl'

USAGE:
    # Simply run it on command line:
    python model.py

    # (You may modify the datafile name/location in '__main__' below)
'''

from get_data import get_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split
import cPickle as pickle
import grid_dict as gd
import matplotlib.pyplot as plt



def tune_models(X_train, y_train):
    '''
    returns a list of fitted models, tuned with optimal hyperparameters
    '''

    (models, scores) = gd.run_grid_search('f1', X_train, y_train, printing=True)
    return (models, scores)



def select_top_model(models, scores, X_test, y_test):
    '''
    returns the highest performing model from a list of models
    '''

    top_model = models[np.argmax(scores)]

    #prints model scores for a variety of metrics
    metrics = [precision_score, recall_score, roc_auc_score, f1_score]
    print top_model.__class__.__name__
    for metric in metrics:
        print "  ",metric.__name__, ":", metric(y_test, top_model.predict(X_test))

    return top_model



def plot_feature_importances(model):
    '''
    if using a Tree-Based model, this produces the feature-importance graph
    '''

    features = ['num_previous_payouts', 'name_length', 'org_desc_length',
                'org_name_length', 'payee_name_empty_string', 'payee_name_length',
                'venue_address_empty_string', 'venue_name_is_null', 'venue_latitude_isnull',
                'org_facebook', 'num_tickets_issued', 'avg_ticket_price', 'max_ticket_price',
                'body_length','user_age', 'name_perc_capital_chars','us_vs_non_us_country',
                'user2event_timediff', 'common_email_domain', 'payout_type_is_empty_string']

    colnames = np.array(features)

    feat_import = model.feature_importances_

    top10_nx = np.argsort(feat_import)[::-1][0:10]

    feat_import = feat_import[top10_nx]
    #normalize:
    feat_import = feat_import /feat_import.max()
    colnames = colnames[top10_nx]

    x_ind = np.arange(10)

    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, colnames[x_ind])
    plt.tight_layout()
    #plt.show()
    plt.savefig('feature_importances.png')


def is_Tree_Based_Model(modelname):
    return if modelname in ['GradientBoostingClassifier', 'AdaBoostClassifier', 'RandomForestClassifier','DecisionTreeClassifier']

def pickle_model(model):
    pickle_file = 'model.pkl'

    with open(pickle_file, 'w') as f:
        pickle.dump(model, f)



if __name__ == '__main__':

    #reads in data (cleans and transforms)
    print "reading training data ..."
    training_data_file = '../data/subset_500.json'
    X, y = get_data(training_data_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    #tunes candidate models
    print "tuning models ..."
    models, scores = tune_models(X_train, y_train)

    #selects top performing model
    top_model = select_top_model(models, scores, X_test, y_test)

    #plots feature importance
    if is_Tree_Based_Model(top_model.__class__.__name__):
        plot_feature_importances(rfc)

    #pickles model
    print "pickling top model ..."
    pickle_model(top_model)
