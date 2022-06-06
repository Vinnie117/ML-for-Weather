from joblib import dump, load

naive_reg = load(r'A:\Projects\ML-for-Weather\models\naive_reg.joblib') 

# The model
print("Coefficients: \n", naive_reg.coef_)



print('END')