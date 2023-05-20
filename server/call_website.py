#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install image_utils


# In[2]:


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
import pandas as pd
import tensorflow as tf

app = Flask(__name__, static_url_path='/static')


# In[3]:


# Load the trained generator model
current_directory = os.path.dirname(__file__)

generator = load_model('os.path.join(current_directory, "generator_model.h5")')  # Replace with the path to your generator model
# Load the pre-trained model
model_path = ('os.path.join(current_directory, "male_to_female.h5")')  # Update with the actual path to your model
fmodel_path = ('os.path.join(current_directory, "female_to_male.h5")')  # Update with the actual path to your model

model = tf.keras.models.load_model(model_path)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust optimizer, loss, and metrics as needed


# In[4]:


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


# In[5]:


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
    


# In[6]:


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


# In[7]:


# Define a function to process the input image
def transform_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)  # Implement the preprocessing steps for your model

    # Perform the gender transformation
    transformed_image = model.predict(preprocessed_image)

    # Postprocess the transformed image if needed
    postprocessed_image = postprocess_image(transformed_image)  # Implement the postprocessing steps for your model

    return postprocessed_image



# def process_image(image : np.ndarray, bounding_box : dlib.rectangle) -> np.ndarray:
#    """IMAGE PREPROCESSING FUNCTION:\nExpand the bounding box such that it includes the hair, chin and cheekbones, and normalize the image""" 
#    rescale = (1.1, 1.9, 1.1, 1.2)

#    (x1, y1, x2, y2) = (bounding_box.left(), bounding_box.top(), bounding_box.right(), bounding_box.bottom())

#    # Expand the bounding boxes according to the rescale factor, take the edge of the image if bounding box ends up expanding beyond it
#    nx1 = max(0, int(x1 - ((x2 - x1) * (rescale[0] - 1) / 2)))
#    ny1 = max(0, int(y1 - ((y2 - y1) * (rescale[1] - 1) / 2)))
#    nx2 = min(image.shape[1], int(x2 + ((x2 - x1) * (rescale[2] - 1) / 2)))
#    ny2 = min(image.shape[0], int(y2 + ((y2 - y1) * (rescale[3] - 1) / 2)))

#    bounding_box = (nx1, ny1, nx2, ny2)

#    # Crop the image into the bounding box
#    image = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

#    # Normalize the facial image into a size of width=96 and height=128 and return it 
#    return cv2.resize(image, (96, 128), interpolation = cv2.INTER_AREA)

def normalize_image(image):
    image = image.astype(np.float32)
    image = (image / 255.0) * 2.0 - 1.0
    image = image[..., ::-1]  # Reverse channel order (BGR to RGB)
    return image

def denormalize(image: np.ndarray) -> np.ndarray:
    """Convert the image from [-1,1] to [0,1] range for displaying."""
    return (image + 1) / 2.0

def generate_images(model: tf.keras.Model, test_input: np.ndarray):
    # the generator uses tanh for the last layer, so we need to reshape
    # the images to be from -1 to 1, hence the preprocessing step
    
    # Preprocess the test input
    test_input = np.expand_dims(test_input, axis=0)  # Add extra dimension

  
    prediction = model(test_input, training=True)
    
    # postprocess the images to be from 0 to 1
    test_input = denormalize(test_input)
    prediction = denormalize(prediction)
    
    plt.figure(figsize=(6, 6))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
    
    return prediction

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    image = normalize_image(image)
    
    # Process the image using the modified generate_images function
    predicted_image = generate_images(model, image)
    
    # Convert the predicted image to a base64 string
    predicted_image_data = predicted_image[0].numpy()  # Convert to NumPy array
    predicted_image_data = (predicted_image_data * 255).astype(np.uint8)
    predicted_image_pil = Image.fromarray(predicted_image_data)
    predicted_image_buffer = io.BytesIO()
    predicted_image_pil.save(predicted_image_buffer, format='PNG')
    predicted_image_base64 = base64.b64encode(predicted_image_buffer.getvalue()).decode('utf-8')

    # Prepare the response data as a JSON object
    response_data = {
        'status': 'success',
        'message': 'Image processed successfully',
        'predicted_image': predicted_image_base64
    }

    return jsonify(response_data)




# In[8]:


@app.route('/fprocess', methods=['POST'])
def fprocess():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    image = normalize_image(image)
    
    # Process the image using the modified generate_images function
    predicted_image = generate_images(model, image)
    
    # Convert the predicted image to a base64 string
    predicted_image_data = predicted_image[0].numpy()  # Convert to NumPy array
    predicted_image_data = (predicted_image_data * 255).astype(np.uint8)
    predicted_image_pil = Image.fromarray(predicted_image_data)
    predicted_image_buffer = io.BytesIO()
    predicted_image_pil.save(predicted_image_buffer, format='PNG')
    predicted_image_base64 = base64.b64encode(predicted_image_buffer.getvalue()).decode('utf-8')

    # Prepare the response data as a JSON object
    response_data = {
        'status': 'success',
        'message': 'Image processed successfully',
        'predicted_image': predicted_image_base64
    }

    return jsonify(response_data)




# In[ ]:


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.run(host='localhost', port=5000)


# In[ ]:





# In[ ]:





# In[ ]:




