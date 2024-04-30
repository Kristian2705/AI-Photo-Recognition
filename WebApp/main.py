import timm as timm
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import keras
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

app = Flask(__name__)

# Folder where the photos will be uploaded
UPLOAD_FOLDER = 'image_store'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained human AI model
model = timm.create_model('deit_base_patch16_224', pretrained=True)
model.head = nn.Sequential(
    nn.Linear(model.head.in_features, 672),
    nn.ReLU(),
    nn.Linear(672, 2)
)
model_path = 'static/models/face_model_0 (1).h5'  # Update with the path to your PyTorch model file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load the trained object model
object_model = keras.models.load_model('static/models/ai_model.h5')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the home page
@app.route('/services')
def services():
    return render_template('service.html')

# Route for the service page
@app.route('/why')
def why():
    return render_template('why.html')

# Route for the team page
@app.route('/team')
def team():
    return render_template('team.html')

# Route for the human model page
@app.route('/human')
def human():
    return render_template('human_model_page.html', result=None)

# Route for the object model page
@app.route('/object')
def object():
    return render_template('object_model_page.html', result=None)

# Logic which processes an uploaded image and returns a result to the route below
@app.route('/upload_human', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        IMAGE_SIZE_HUMAN = (224, 224)

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = Image.open(filename)

        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        random_affine = transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1),
                                                shear=(-10, 10))

        add_noise = AddGaussianNoise(0., 0.1)
        random_erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_HUMAN),
            color_jitter,
            random_affine,
            transforms.ToTensor(),
            add_noise,
            random_erasing,
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            # Normalize using ImageNet defaults
        ])

        transformed_image = transform(img)
        transformed_image = transformed_image.unsqueeze(0)
        y_logit = model(transformed_image)
        pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        result = "AI-generated" if pred.item() == 0 else "Real"
        response = render_template('human_model_page.html', result=result)
        return response

@app.route('/upload_object', methods=['POST'])
def upload_object_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        IMAGE_SIZE_OBJECT = (256, 256)

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Create an Image Object from an Image
        img = Image.open(filename)

        # Make the uploaded image have the desired size
        resized_im = img.resize(IMAGE_SIZE_OBJECT)

        # Convert the image to a NumPy array
        img_array = image.img_to_array(resized_im)

        # Expand the dimensions to match the input shape expected by the model
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction using the model
        prediction = object_model.predict(img_array)
        result = "AI-generated" if prediction[0][0] < 0.5 else "Real"
        response = render_template('object_model_page.html', result=result)
        return response

if __name__ == '__main__':
    app.run()
