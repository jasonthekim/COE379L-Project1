from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load_model("./models/Alternate_Lenet5.h5")

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

# Define the image size expected by the model
img_height = 128
img_width = 128

# Define the batch size for inference
batch_size = 32

# Define the rescaling function for the dataset
datagen = ImageDataGenerator(rescale=1./255)

@app.route('/predict_and_evaluate_dataset', methods=['POST'])
def predict_and_evaluate_dataset():
    try:
        # Check if the request contains the dataset directory path in the URL parameters
        dataset_dir = request.args.get('dataset_dir')
        print("Dataset directory:", dataset_dir)  # Debugging statement
        if not dataset_dir:
            return jsonify({'error': 'No dataset directory provided in the request URL parameters'}), 400

        # Load the dataset using the ImageDataGenerator
        print("Loading dataset from directory:", dataset_dir)  # Debugging statement
        dataset = datagen.flow_from_directory(
            dataset_dir,  # Use the provided directory path
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',  # Assuming it's a binary classification problem
            shuffle=False  # Ensure that images are in the same order as predictions
        )

        # Predict probabilities for each image in the dataset
        print("Predicting probabilities for dataset")  # Debugging statement
        predictions = model.predict(dataset)

        # Evaluate the model's performance on the dataset
        print("Evaluating model performance on dataset")  # Debugging statement
        test_loss, test_accuracy = model.evaluate(dataset, verbose=0)

        # Convert the prediction results and evaluation metrics into a JSON format
        results = {
            'predictions': predictions.tolist(),  # Convert to list for JSON serialization
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

        # Return the JSON response
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)