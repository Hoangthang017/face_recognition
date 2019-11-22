from flask import Flask, render_template, Response,url_for, jsonify, request, redirect,json,flash,session
import face
import os
import cv2

from werkzeug.utils import secure_filename



app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\Users\\WIND\\OneDrive\\Project\\vn_celeb_face_recognition\\FaceRecognitionIdol\\static\\images\\images_update'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/name', methods=['GET', 'POST'])
def upload_image():
    #file=request.files['image']
    #file=request.files.get('file')
    file = request.files['image']
    filename = secure_filename(file.filename)
    if request.method == 'POST':
   
        fileIMG=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fileIMG)
        faceName = face.detect_faces_in_image(fileIMG)
        os.remove(fileIMG)
            
        return render_template("nameidol.html",name=faceName)
    


    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))




if __name__ == '__main__':

    app.run(debug=1)
    