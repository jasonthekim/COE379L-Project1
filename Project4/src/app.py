from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from transformers import CLIPProcessor, CLIPModel
import io

app = Flask(__name__)

# Load your trained model (assuming it's saved under 'model.pt')
model = torch.load('./clip_binary_model.pth')
model.eval()

# Initialize the CLIP processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect_ratio(img, 256)),
        transforms.CenterCrop(224)
    ])
    return transform(image)

def resize_with_aspect_ratio(image, desired_size):
    old_width, old_height = image.size
    aspect_ratio = old_width / old_height
    if old_width >= old_height:
        new_width = desired_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = desired_size
        new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height), Image.LANCZOS)

# Define the text preprocessing
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english') and token not in string.punctuation]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Missing image or text'}), 400

    # Image processing
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    transformed_image = transform_image(image)

    # Text processing
    text = request.form['text']
    processed_text = preprocess_text(text)

    # Prepare the inputs for the model
    inputs = processor(text=processed_text, images=transformed_image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Assuming the model outputs logits and you apply sigmoid to get probabilities
    probabilities = torch.sigmoid(outputs.logits)
    prediction = 'Non-offensive' if probabilities < 0.5 else 'Offensive'

    return jsonify({'prediction': prediction, 'probability': probabilities.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
