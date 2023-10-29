from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)

# Implement model
from model_training import model

data_predictions = []  # To store incoming data and predictions


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        data_point = [[json_['temperature'], json_['humidity'], json_['sound_volume']]]

        scores = model.decision_function(data_point)


        # Convert NumPy int32 to Python int
        anomaly_score = float(scores)

        # Store the data and prediction
        data_predictions.append({'data': json_, 'anomaly_score': anomaly_score})

        # Return the prediction and anomaly score
        return jsonify({'anomaly_score': anomaly_score})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', data_predictions=data_predictions)


if __name__ == '__main__':
    app.run(port=5000)

