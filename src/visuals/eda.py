######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df
from models.naive_model import y_test


x = pd_df['temperature_lag_1']
y = pd_df['temperature']

plt.scatter(x, y, s=1)

plt.title("Lag Analysis")
plt.xlabel("Temperature (°C) at time t-1")
plt.ylabel("Temperature (°C) at time t-1")

plt.show()

print('END')