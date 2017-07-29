'''
DESCRIPTION:
    - Loads the predictive model (saved as pickle in model.py)
    - Captures a new Event datapoint from the Events streaming server
    - Calculates probability that Event is a Fraud
    - Stores fraud detection results in a database


USAGE:
    # Simply run it on command line:
    python predict.py
'''


import cPickle as pickle
import json
from get_data import get_x_matrix_from_file
#from persistence import update_events_fraud_record, persist_record_to_events_mongo
from pymongo import MongoClient
import requests



def unpickle_model():
    '''
    unpickles and returns a pre-trained prediction model
    '''
    pickle_file = 'model.pkl'

    with open(pickle_file) as f:
        return pickle.load(f)



def predict_fraud(X_test, model):
    '''
    predicts whether given data point is fraudulent, based on the given model.
    returns a probability value (0.0 - 1.0) that data point is a Fraud.
    '''

    return model.predict_proba(X_test)[0][1]

def update_doc_with_prediction(document, fraud_prob):
    '''
    inserts a data point and it's predicted target value into a database
    '''
    update_events_fraud_record(document, fraud_prob)


def request_new_event():
    '''
    retrieves new (unseen) event datapoint from a web server.

    NOTE: actual code removed per confidentiality agreement.
          replaced it with a read of a single event from a local file
    '''

    ### SENSITIVE DETAILS REMOVED ###

    ## created from: tail -1  ../data/subset.json > ../data/test_script_examples
    #file must only contain 1 json entry
    test_data_file = '../data/test_script_examples'

    return get_x_matrix_from_file(test_data_file)


if __name__ == '__main__':

    # unpickles model that was generated and saved in model.py
    model = unpickle_model()

    # reads in a single test data point
    X_test = request_new_event()

    # makes prediction on test data point
    fraud_prob = predict_fraud(X_test, model)

    # prints the probability that the Test event is Fraudulent
    print "probability of this event being fraud is: ", fraud_prob, "\n"


    # the following saves each new event datapoint, and it's prediction Fraud/Not Fraud porbaility into mongo DB
    # NOTE: turning this off for now - until we turn on MongoDB for live web demonstration
    # client = MongoClient('mongodb://localhost:27017/')
    # update_doc_with_prediction(event, fraud_prob)
