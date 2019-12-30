import os
import pickle
import time
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import asarray
from numpy import expand_dims
from PIL import Image
from matplotlib import pyplot
from matplotlib import colors
from scipy.spatial.distance import cosine
from keras.models import Sequential
from keras.layers import Flatten
from keras import Model
import vptree
def cosine_similarity(p1, p2):
    return cosine(p1, p2)
# pickle_in_dic = open('embeded_face_train_resnet50_vptree.pickle', 'rb')
# dic = pickle.load(pickle_in_dic)
# pickle_in_dic.close()
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
graph = tf.get_default_graph()
def resize_image(filename, required_size=(224, 224)):
    # pixels = pyplot.imread(filename)
    # image = Image.fromarray(pixels)
    image = Image.open(filename)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    pixels = pixels[:,:,:3]
    print('dtype', pixels.dtype) 
    if pixels.dtype == np.float64 or pixels.dtype == np.float32:
        pixels *= 255
        pixels = pixels.astype(np.uint8)
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_embeddings(filenames, need_to_extract=False, model=model):
    global graph
    with graph.as_default():
        # extract face
        faces = []
        if need_to_extract is False:
            faces = [resize_image(f) for f in filenames]
        else:
            faces = [extract_face(f) for f in filenames]  
        # convert into an array of samples
        samples = asarray(faces, 'float32')
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)
        # perform prediction
        yhat = model.predict(samples)
        return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        # print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        return True
    # else:
        # print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return False

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def retrieveFaces(query_img, data_dir='embeded_face.pickle'):
    pickle_in = open(data_dir, 'rb')
    embeddings = pickle.load(pickle_in)
    pickle_in.close()
    query_embedding = get_embeddings([query_img], need_to_extract=True)
    query_embedding = query_embedding[0]
    retrieval_result = []
    for i in range(len(embeddings)):
        if is_match(query_embedding, embeddings[i][0], 0.3) is True:
            retrieval_result.append(embeddings[i][1])
    return retrieval_result

def retreive_top_k(query_img, k=1):
    query_embedding = get_embeddings([query_img], need_to_extract=True)
    query_embedding = query_embedding[0]
    return dic['tree'].get_n_nearest_neighbors(query_embedding, k)