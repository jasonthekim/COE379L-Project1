from flask import Flask, request, jsonify
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from transformers import CLIPProcessor, CLIPModel
import io

app = Flask(__name__)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class CLIPBinaryModel(nn.Module):
    def __init__(self, clip_model, feature_dim=1024):
        super(CLIPBinaryModel, self).__init__()
        self.clip_model = clip_model
        self.binary_classifier = BinaryClassifier(input_dim=feature_dim)

    def forward(self, input_ids, attention_mask, pixel_values):
        self.clip_model.eval()  # Ensure the CLIP model is in eval mode
        with torch.no_grad():
            outputs = self.clip_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        features = torch.cat((image_features, text_features), dim=1)
        return self.binary_classifier(features)

# Load your trained model (assuming it's saved under 'clip_binary_model.pth')
model = CLIPBinaryModel(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))

# Load the state dictionary into the model
model_state_dict = torch.load('./clip_binary_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict.state_dict())
model.eval()

# Initialize the CLIP processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.CenterCrop(224)  # Crop to 224x224
    ])
    return transform(image)

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
    probabilities = torch.sigmoid(outputs)
    prediction = 'Non-offensive' if probabilities < 0.5 else 'Offensive'

    return jsonify({'prediction': prediction, 'probability': probabilities.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
