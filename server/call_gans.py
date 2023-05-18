#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import base64
import io
import os
import cv2
from PIL import Image
import base64

app = Flask(__name__, static_url_path='/static')

# Load the trained generator model
generator = load_model('C:/Users/BenLXH/Desktop/FYP/FACE/generator_model.h5')  # Replace with the path to your generator model

@app.route('/')
def index():
    return render_template('try.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Generate a new image
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)

    # Convert the image to base64 string
    image_data = generated_image[0, :, :, 0]
    image_data = (image_data * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the image as a JSON response
    return jsonify(image_data=image_base64)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        filename = image.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        # You can perform additional operations on the uploaded image if needed
        thumbnail_size = (200, 200)  # Define the size of the thumbnail
        thumbnail_filename = 'thumbnail_' + filename
        thumbnail_filepath = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)
        img = Image.open(filepath)
        img.thumbnail(thumbnail_size)
        img.save(thumbnail_filepath)
        # Finally, return the URL or any relevant information about the uploaded image
        return f'/uploads/{filename}'
    else:
        return 'No image file provided.'
    
@app.route('/process_image', methods=['POST'])
def process_image():
    # Retrieve the image data from the request
    image_data = request.files['imageData'].read()

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = 255 - gray

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0, 0)

    # Create the sketch by dividing grayscale image by the inverse blurred image
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # Convert the sketch to base64 image data
    _, processed_image_data = cv2.imencode('.png', sketch)
    processed_image_data = processed_image_data.tobytes()
    processed_image_data = base64.b64encode(processed_image_data).decode('utf-8')

    # Return the processed image data
    return processed_image_data


# In[ ]:


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.run(host='localhost', port=5000)


# In[ ]:




