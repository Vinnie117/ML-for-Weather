import os, logging


def save(var, train, test, train_std, test_std):
    ''' Save (standardized) training and test data to folder ./data_dvc/processed
    @param target: The target variable of interest
    @param train: the training data to be saved
    @param test: the test data to be saved
    
    '''

    logging.info('SAVING DATA TO DIRECTORY')

    dir_name = os.path.join(os.getcwd(), 'data_dvc', 'processed') 
    base_filename_train = 'train_' + var
    base_filename_test = 'test_' + var
    base_filename_train_std = 'train_std_' + var
    base_filename_test_std = 'test_std_' + var
    format = 'csv'

    file_train = os.path.join(dir_name, base_filename_train + '.' + format)
    file_test = os.path.join(dir_name, base_filename_test + '.' + format)
    file_train_std = os.path.join(dir_name, base_filename_train_std + '.' + format)
    file_test_std = os.path.join(dir_name, base_filename_test_std + '.' + format)

    train.to_csv(file_train, header=True, index=False)
    test.to_csv(file_test, header=True, index=False)
    train_std.to_csv(file_train_std, header=True, index=False)
    test_std.to_csv(file_test_std, header=True, index=False)


def dict_to_df(dict_data):
    '''
    This function creates the complete dataframes after preprocessing. It specifically
    works with a dict of dicts containing data folds
    '''
    logging.info('FETCH DATAFRAMES FROM DICTIONARY')

    # the last fold is complete data
    last_train_key = list(dict_data['train'])[-1]
    last_test_key = list(dict_data['test'])[-1] 

    # full dataframes
    train = dict_data['train'][last_train_key]
    test = dict_data['test'][last_test_key]
    train_std = dict_data['train_std'][last_train_key]
    test_std = dict_data['test_std'][last_test_key]

    return train, test, train_std, test_std
