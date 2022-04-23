import sys 
sys.path.append('A:\Projects\ML-for-Weather\src')  # import from parent directory
import matplotlib.pyplot as plt
from features.pipeline_dataprep import pd_df
from models.naive_model import y_test, y_pred_test


x = pd_df['timestamp'].tail(len(y_pred_test))

plt.plot(x, y_test, label = "actual", alpha = 0.5)
plt.plot(x, y_pred_test, label = "predicted", alpha = 0.5)

plt.title("Evaluation of Prediction")
plt.xlabel("Date")
plt.ylabel("Temperature in °C")

plt.legend()
plt.show()


print("END")