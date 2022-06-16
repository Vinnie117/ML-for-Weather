import sys
sys.path.append('A:\Projects\ML-for-Weather\src') 
from features.pipeline_dataprep import pd_df, data
from features.pipeline_dataprep import train_folds, test_folds, train_std_folds, test_std_folds

# the last fold -> complete data. Last key is same for raw and std. data
last_train_key = list(data['train'])[-1]
last_test_key = list(data['test'])[-1] 

# full dataframes
train = data['train'][last_train_key]
test = data['test'][last_test_key]
train_std = data['train_std'][last_train_key]
test_std = data['test_std'][last_test_key]
pd_df = data['pd_df'] # train + test

# sub folds
train_fold_0 = data['train']['train_fold_0']
train_std_fold_0 = data['train_std']['train_fold_0'] 
test_fold_0 = data['test']['test_fold_0']
test_std_fold_0 = data['test_std']['test_fold_0']

# Std. parameters
mean_train = train['temperature_lag_1'].mean()
std_train = train['temperature_lag_1'].std()
print('Mean for z-score is:', mean_train)
print('Stddev for z-score is:', std_train)


print( (5.8 - mean_train)/std_train)         # -0.879333  


###################################################################
# Testing  data
# info: https://stackoverflow.com/questions/5997027/python-rounding-error-with-float-numbers

print(train)
# Check: Was training data standardized correctly?
z1 = (7.6 - mean_train)/std_train    
print(z1)                                 # -0.6415251952932665                    
print(train_std)

print(train_fold_0)                       # the values to be standardized
z2 = (8.3 - mean_train)/std_train         # one standardized value
print(z2)                                 # -0.5488323209229191
print(train_std_fold_0)                   # the standardized values

# Check: Was test data standardized correctly?
print(test)
z3 = (0.9 - mean_train)/std_train
print(z3)                                  # -1.5287284214094465
print(test_std)

print(test_fold_0)
z4 = (9.8 - mean_train)/std_train
print(z4)                                  # -0.3502047329864609
print(test_std_fold_0)
          


print('END')

