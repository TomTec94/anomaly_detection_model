import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Test size
num_points = 1000

# Normal data
temperatures = np.random.uniform(20, 30, num_points)
humidity = np.random.uniform(30, 70, num_points)
sound_volume = np.random.uniform(40, 90, num_points)

# Create array
normal_data = np.column_stack((temperatures, humidity, sound_volume))

# Anomaly data
num_anomalies = 50
anomalous_temperatures = np.random.uniform(10, 150, num_anomalies)
anomalous_humidity = np.random.uniform(60, 100, num_anomalies)
anomalous_sound_volume = np.random.uniform(20, 150, num_anomalies)

# Create array
anomalous_data = np.column_stack((anomalous_temperatures, anomalous_humidity, anomalous_sound_volume))

# Join data
data = np.vstack((normal_data, anomalous_data))

# Train the model
model = IsolationForest()
model.fit(data)

# Get anomaly scores
scores = model.decision_function(data)

# Create histogram
plt.figure(figsize=(10, 7))
plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

