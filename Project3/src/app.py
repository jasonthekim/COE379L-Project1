from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model("./models/Alternate_Lenet5.keras")

# Helper functions
def extract_model_info(model):
    info = {
        "name": "Alternate-Lenet-5",
        "description": "Best performing model that classifies images as containing buildings that are either damaged or not damaged",
        "parameters_count": model.count_params()
    }
    return info

def get_model_summary_table():
    # Redirect summary to a string buffer
    import sys
    import io
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Get the model summary
    model.summary()

    # Get the model summary string
    summary_string = new_stdout.getvalue()
    
    # Reset stdout
    sys.stdout = old_stdout

    return summary_string

# Endpoints
@app.route('/model_summary', methods=['GET'])
def model_summary2():
    model_info = extract_model_info(model)
    return jsonify(model_info)

@app.route('/model_summary_table', methods=['GET'])
def model_summary_table():
    # Get the model summary as a string
    summary = get_model_summary_table()
    
    # Include metadata
    metadata = {
        'model_name': 'Alternate_Lenet5'
    }

    # Convert metadata to string
    metadata_str = ', '.join(f"{key}: {value}" for key, value in metadata.items())
    
    # Combine metadata and model summary
    result = metadata_str + '\n\n' + summary
    
    # Return the result
    return result

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Check if the request contains the image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided in the request'}), 400

        # Read the image file from the request
        img_file = request.files['image']

        # Open the image using PIL
        img = Image.open(img_file)
        
        # Resize and preprocess the image
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform inference using the pre-trained model
        prediction = model.predict(img_array)

        # Extract the probability of being damaged
        probability_damaged = prediction[0][0]

        # Determine the class based on the threshold
        threshold = 0.5
        if probability_damaged >= threshold:
            prediction_label = 'No Damage'
        else:
            prediction_label = 'Damaged'

        # Convert the prediction result into a JSON format
        result = {
            'probability_no_damage': float(probability_damaged),
            'prediction': prediction_label
        }

        # Return the JSON response
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)