import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Upload folder configuration (inside static directory)
UPLOAD_FOLDER = r'X:\flask_app_MI.AI\static\upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = tf.keras.models.load_model(r'X:\flask_app_MI.AI\model\best_model.h5')


def process_image(image_path):
    """Process the uploaded image to prepare it for the model."""
    img = Image.open(image_path).convert('RGB')  # Convert to RGB (3 channels)
    img = img.resize((224, 224))  # Resize to match the model's expected input size
    img_array = np.array(img) / 255.0  # Normalize to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension
    return img_array


@app.route('/')
def index():
    return render_template('index.html')  # Renders index.html

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, process it, and return the prediction."""
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400
    # Save the uploaded file in static/uploads
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    # Process the image and predict
    processed_image = process_image(file_path)
    prediction = model.predict(processed_image)
    print(prediction)
    Class_Names = ['Aster', 'Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']
    prediction_label = Class_Names[np.argmax(prediction, axis=1)[0]]


    return render_template('prediction.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
