from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import urllib.request
import os
from werkzeug.utils import secure_filename
from predict import *
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')

def home():
    return render_template('index.html')


@app.route('/', methods = ["POST"])
def upload_image1():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    file = request.files.getlist('file')
     
    errors = {}
    success = False
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        #flash('Image successfully uploaded and displayed below')
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result = detect(path)
        if result == 0:
            return "Couldn't detect the leaf"
        else:
            return result
        #file.save(os.path.join('static//output//', op_filename))
        #return os.path.join('static//output//', op_filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return "No Image"
 
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('file')
     
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed below')
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result = detect(path)
            if result == 0:
                resp = jsonify({'message' : "Could Not Detect Leaf"})
                resp.status_code = 201
                return resp
            else:
                resp = jsonify({'message' : result})
                resp.status_code = 201
                return resp
                #return result
            #file.save(os.path.join('static//output//', op_filename))
            #return os.path.join('static//output//', op_filename)
        else:
            #flash('Allowed image types are - png, jpg, jpeg, gif')
            resp = jsonify({'message' : "No Image"})
            resp.status_code = 201
            return resp
 
 
if __name__ == "__main__":
    app.run()