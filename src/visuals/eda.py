######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test
#from models.naive_model import y_test


################################################################

x_scaled = test['temperature_lag_1']
y_scaled = test['target_temperature']

plt.scatter(x_scaled, y_scaled, s=1)

plt.title("Lag Analysis")
plt.xlabel("Scaled Temperature (°C) at time t-1")
plt.ylabel("Temperature (°C) at time t")

plt.show()

print('END')