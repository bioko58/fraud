'''
DESCRIPTION:
    Transforms raw Events data into features for predictive modeling.

    Reads the Events data, performs feature engineering, and returns a
    feature matrix (DataFrame), and class labels Fraud/Not-Fraud (np.array)
    for all the given Events data


USAGE:
    # For labeled (supervised) train/test data:
    # import and call get_data() from within modeling file (e.g. model.py):

    from get_data import get_data
    X, Y = get_data(path/to/filename)


    # For unlabeled data (live/unsupervised data w/no actual label):
    # import and call get_x_matrix_from_file(filename):

    from get_data import get_x_matrix_from_file
    X_test = get_x_matrix_from_file(path/to/filename)


    # There's need to run this script directly / standalone.
    # But if you still want to, you'll get the Feature Matrix (X), and
    # Class Labels (Y) returned on the CLI. see __main__
'''


import pandas as pd
import json
import numpy as np


### FEATURE ENGINEERING ###

def proportion_capital_characters(s):
    '''
    input: takes in a string the account name
    output: returns a float of the proportion of capitial letters in the name
    '''
    s_len = len(s)
    num_capital_cars = 0
    for c in s:
       if c.isupper():
           num_capital_cars += 1
    if s_len > 0:
       return float(num_capital_cars) / s_len
    else:
       return 0

def create_number_tickets_issued(ticket_types):
    '''
    input: takes in a
    output: returns the total number of tickets offered for one account
    '''
    num_tickets = 0
    for rec in ticket_types:
       num_tickets += rec['quantity_total']
    return num_tickets

def create_average_ticket_price(ticket_types):
    '''
    input: takes in a
    output: returns the average ticket price for one account
    '''
    num_tickets = 0
    amount = 0
    for rec in ticket_types:
       num_tickets += rec['quantity_total']
       amount += rec['cost'] * rec['quantity_total']
    if num_tickets > 0 :
       avg_ticket_price = float(amount) / num_tickets
    else:
       avg_ticket_price = 0.0
    return avg_ticket_price

# def create_stddev_ticket_price(ticket_types):
#     '''
#     input: takes in a
#     output: returns the standard deviation between the ticket prices for one account
#     '''
#     amounts = []
#     for rec in ticket_types:
#        amounts.append(rec['cost'])
#     return np.std(amounts)

def create_max_ticket_price(ticket_types):
    '''
    input: takes in a
    output: returns the max ticket price for 1 account
    '''
    amounts = []
    for rec in ticket_types:
       amounts.append(rec['cost'])
    if len(amounts) > 0:
       max_cost = np.max(amounts)
    else:
       max_cost = 0
    return max_cost

def check_payout_type_is_empty_string(df):
    '''
    input: dataframe
    output: return pandas series with a 1 if the payout type is empty, 0 otherwise
    '''
    df['payout_type_is_empty_string'] = df.payout_type.map(lambda x: 1 if x == '' else 0)
    return df['payout_type_is_empty_string']

def common_email_domain(df):
    '''
    input: dataframe
    output: return pandas series with a 1 if the email is from common domain, 0 otherwise
    '''
    df['common_email_domain'] = df.email_domain.map(lambda x: 1 if (x.split('.')[0] in {'gmail', 'yahoo', 'live', 'hotmail', 'msn'}) else 0)
    return df['common_email_domain']

def userevent_timediff(df):
    '''
    input: dataframe
    output: return pandas series with an integer of the number of days between when
            the even was created and when the user account was created
    '''
    user2event_timediff = (df.event_created - df.user_created).dt.days
    df['user2event_timediff'] = user2event_timediff
    return df['user2event_timediff']

def us_or_nonus(df):
    '''
    input: dataframe
    output: return pandas series with 1 if the country is US, 0 otherwise
    '''
    df['us_vs_non_us_country'] = df.country.map(lambda x: 1 if x == 'US' else 0)
    return df['us_vs_non_us_country']

def create_proportion_capital_characters(df):
    '''
    input: dataframe
    output: returns pandas series of the proportion of capital characters in the name
    '''
    df['name_perc_capital_chars'] = df.name.map(lambda x: proportion_capital_characters(x))
    return df['name_perc_capital_chars']

def num_tickets_issued(df):
    '''
    input: dataframe
    output: pandas series with the number of tickets issued per account
    '''
    df['num_tickets_issued'] = df.ticket_types.map(lambda x: create_number_tickets_issued(x))
    return df['num_tickets_issued']

def avg_ticket_price(df):
    '''
    input: dataframe
    output: panda series with the average ticket prices per account
    '''
    df['avg_ticket_price'] = df.ticket_types.map(lambda x: create_average_ticket_price(x))
    return df['avg_ticket_price']

# def stddev_ticket_price(df):
#     '''
#     input: dataframe
#     output: pandas series that returns the standard deviation between the
#             offered ticket prices per account
#     '''
#     df['stddev_ticket_price'] = df.ticket_types.map(lambda x: create_stddev_ticket_price(x))
#     return df['stddev_ticket_price']

def max_ticket_price(df):
    '''
    input: dataframe
    output: pandas series with the max ticket price offered per account
    '''
    df['max_ticket_price'] = df.ticket_types.map(lambda x: create_max_ticket_price(x))
    return df['max_ticket_price']

def fill_org_facebook_mode(df):
    '''
    input: dataframe
    output: pandas series with the null values replaces with 0
    '''
    df['org_facebook'] = df.org_facebook.map(lambda x: 0 if pd.isnull(x) else x)
    return df['org_facebook']

def create_venue_latitude_isnull(df):
    '''
    input: dataframe
    output: pandas series with a 1 if the venue latitude is null, 0 otherwise
    '''
    df['venue_latitude_isnull'] = df.venue_latitude.map(lambda x: 1 if pd.isnull(x) else 0)
    return df['venue_latitude_isnull']

def create_venue_name_isnull(df):
    '''
    input: dataframe
    output: pandas series with a 1 if the venue name is null, 0 otherwise
    '''
    df['venue_name_is_null'] = df.venue_name.map(lambda x: 1 if pd.isnull(x) else 0)
    return df['venue_name_is_null']

def create_venue_address_empty_string(df):
    '''
    input: dataframe
    output: pandas series with a 1 if the venue address is an empty sting, 0 otherwise
    '''
    df['venue_address_empty_string'] = df.venue_address.map(lambda x: 1 if x == '' else 0)
    return df['venue_address_empty_string']

def create_payee_name_length(df):
    '''
    input: dataframe
    output: pandas series with the length of the payee name
    '''
    df['payee_name_length'] = df.payee_name.map(lambda x: len(x))
    return df['payee_name_length']

def create_payee_name_empty_string(df):
    '''
    input: dataframe
    output: pandas series with a 1 if the payee name is an empty string, 0 otherwise
    '''
    df['payee_name_empty_string'] = df.payee_name.map(lambda x: 1 if x == '' else 0)
    return df['payee_name_empty_string']

def create_org_name_length(df):
    '''
    input: dataframe
    output: pandas series with the length of the organization name
    '''
    df['org_name_length'] = df.org_name.map(lambda x: len(x))
    return df['org_name_length']

def create_org_desc_length(df):
    '''
    input: dataframe
    output: pandas series with the len of the organization description
    '''
    df['org_desc_length'] = df.org_desc.map(lambda x: len(x))
    return df['org_desc_length']

def create_name_length(df):
    '''
    input: dataframe
    output: pandas series with the length of name
    '''
    df['name_length'] = df.name.map(lambda x: len(x))
    return df['name_length']

def create_num_previous_payouts(df):
    '''
    input: dataframe
    output: pandas series with the number of previous payouts
    '''
    df['num_previous_payouts'] = df.previous_payouts.map(lambda x: len(x))
    return df['num_previous_payouts']

### FEATURE ENGINEERING END ##


def convert_to_time(df):
    df.approx_payout_date = pd.to_datetime(df.approx_payout_date,unit='s')
    df.event_created = pd.to_datetime(df.event_created, unit='s')
    df.event_end = pd.to_datetime(df.event_end, unit='s')
    df.event_published = pd.to_datetime(df.event_published, unit='s')
    df.event_start = pd.to_datetime(df.event_start, unit='s')
    df.user_created = pd.to_datetime(df.user_created, unit='s')

def subset_dataframe(df):
    features = ['num_previous_payouts', 'name_length', 'org_desc_length',
                'org_name_length', 'payee_name_empty_string', 'payee_name_length',
                'venue_address_empty_string', 'venue_name_is_null', 'venue_latitude_isnull',
                'org_facebook', 'num_tickets_issued', 'avg_ticket_price', 'max_ticket_price',
                'body_length','user_age', 'name_perc_capital_chars','us_vs_non_us_country',
                'user2event_timediff', 'common_email_domain', 'payout_type_is_empty_string']

    feature_funcs = [create_num_previous_payouts, create_name_length, create_org_desc_length,
                    create_org_name_length, create_payee_name_empty_string, create_payee_name_length,
                    create_venue_address_empty_string, create_venue_name_isnull, create_venue_latitude_isnull,
                    fill_org_facebook_mode, num_tickets_issued, avg_ticket_price, max_ticket_price,
                    create_proportion_capital_characters, us_or_nonus, userevent_timediff,
                    common_email_domain, check_payout_type_is_empty_string]
    for func in feature_funcs:
        func(df)

    X = df[features]
    return X.values

def get_labels(df):
    '''
    input: dataframe
    output: an array of 1's and 0's corresponding to labels
    '''
    y = df.acct_type.map(lambda x: 1 if x.startswith('fraud') else 0)
    return y.values

def get_x_matrix(data):
    '''
    input: dataframe
    output: returns numpy matrix of feature variables
    '''
    try:
        df = pd.DataFrame.from_dict(data, orient= 'index').T.reset_index()
    except:
        print "FAILED CONVERSION TO DATAFRAME"

    return subset_dataframe(df)


def get_x_matrix_from_file(filename):
    '''
    input: data file
    output: returns numpy matrix of feature variables
    '''
    try:
        df = pd.read_json(filename)
    except:
        with open(filename) as json_data:
            data = json.load(json_data)
        df = pd.DataFrame.from_dict(data, orient= 'index').T.reset_index()
    convert_to_time(df)
    return subset_dataframe(df)

def get_data(filename):
    '''
    input: reads in a json file from a filepath
    output: outputs X (feature matrix) and y (labels)
    '''
    df = pd.read_json(filename)
    convert_to_time(df)
    return subset_dataframe(df), get_labels(df)


if __name__ == '__main__':
    X,y = get_data('data/data.json')
