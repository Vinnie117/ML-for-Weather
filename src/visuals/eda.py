######## Exploratory Data Analysis ########
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'A:\Projects\ML-for-Weather\data\interim\df.csv') 

# matplotlib seems to have trouble loading the plot with so much data?
plt.plot(df["date"], df["cloud_cover_total"])
plt.show()






