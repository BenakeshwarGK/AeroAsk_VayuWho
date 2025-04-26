from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('aeroclassify_model.h5')

# Load aircraft data
aircraft_df = pd.read_csv('aircraftQA.csv')

# Extract aircraft ID list
aircraft_ids = aircraft_df['AircraftID'].unique().tolist()

# Predict function
def predict_aircraft(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    return aircraft_ids[predicted_index]

@app.route('/')
def index():
    options = list(aircraft_df.columns)
    options.remove('AircraftID')  # Don't show ID as a query option
    return render_template('index1.html', options=options)

@app.route('/result', methods=['POST'])
def result():
    file = request.files['file']
    image = Image.open(file.stream)
    query = request.form['query']

    predicted_id = predict_aircraft(image)

    matched_row = aircraft_df[aircraft_df['AircraftID'] == predicted_id]
    if matched_row.empty:
        response = "No data found for this aircraft."
    else:
        response = matched_row[query].values[0]

    return render_template('result.html', aircraft_id=predicted_id, query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)
