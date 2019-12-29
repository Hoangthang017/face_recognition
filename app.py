from flask import Flask, render_template, Response,url_for, jsonify, request, redirect, json,flash, session
import os
import face_retrieval
from werkzeug.utils import secure_filename

app = Flask(__name__)
curr_dir = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(*[curr_dir, 'static', 'images', 'upload'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/name', methods=['GET', 'POST'])
def upload_image():
    file_list = []
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return redirect('/')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_list = face_retrieval.retrieveFaces(query_img=os.path.join(app.config['UPLOAD_FOLDER'], filename), data_dir='embeded_face_train_resnet50.pickle')
    return render_template('nameidol.html', file_list=file_list)

    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=1, threaded=False)
    