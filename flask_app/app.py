from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
# from model import predict_image
from skin_cancer import predict_image as predict_image_skin
from chest_cancer import predict_image as predict_image_chest
import torch
from torch import nn as nn
from model.alexnet import AlexNet





app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    image_type = request.form.get('image_type')
    return redirect("/chest") if image_type == 'Chest X-Ray' else redirect("/skin")

@app.route('/skin', methods=['GET', 'POST'])
def upload_file_skin():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image_skin(filepath)
            os.remove(filepath)  # Optionally remove the file after prediction
            return render_template('skin/result.html', prediction=prediction)
    return render_template('skin/upload.html')

@app.route('/chest', methods=['GET', 'POST'])
def upload_file_chest():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image_chest(filepath)
            os.remove(filepath)  
            return render_template('chest/result.html', prediction=prediction)
    return render_template('chest/upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
