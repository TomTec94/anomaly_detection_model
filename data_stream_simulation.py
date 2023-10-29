import requests
import numpy as np
import time

API_ENDPOINT = "http://127.0.0.1:5000/predict"

def generate_data():
    radom_work = np.random.uniform(0,100)
    if radom_work <=90:
        temperature = np.random.uniform(20, 30)
        humidity = np.random.uniform(30, 70)
        sound_volume = np.random.uniform(40, 90)
    else:
        temperature = np.random.uniform(10, 150)
        humidity = np.random.uniform(60, 100)
        sound_volume = np.random.uniform(20, 150)
    return {'temperature': temperature, 'humidity': humidity, 'sound_volume': sound_volume}

def send_data(data):
    response = requests.post(API_ENDPOINT, json=data)
    return response.json()



def main():
    while True:
        data = generate_data()
        result = send_data(data)
        print("Sent data:", data)
        print("Received prediction:", result)
        time.sleep(5)

if __name__ == "__main__":
    main()

