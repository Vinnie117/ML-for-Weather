######## Exploratory Data Analysis ########
import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df, train, test
#from models.naive_model import y_test


################################################################

x1 = train['temperature_lag_1']
x2 = train['temperature_lag_2']
y = train['temperature']

print(x1.name)

# plt.scatter(x1, y, s=1)
# plt.title("Lag Analysis")
# plt.xlabel("Temperature (°C) at time t-1")
# plt.ylabel("Temperature (°C) at time t")
# plt.show()

# plt.scatter(x2, y, s=1)
# plt.title("Lag Analysis")
# plt.xlabel("Temperature (°C) at time t-2")
# plt.ylabel("Temperature (°C) at time t")
# plt.show()

fig, axs = plt.subplots(2, 2)
fig.suptitle('Lag Analysis')
axs[0, 0].scatter(x1, y, s=1)
#axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].scatter(x2, y, s=1)
#axs[0, 1].set_title('Axis [0, 1]')

# Auf den Namen des der Series x1, x2, ... zugreifen!
for i, ax in enumerate(axs.flat):
    ax.set(xlabel='temperature_lag_{}'.format(i+1), ylabel='temperature')

fig.show()

print('END')