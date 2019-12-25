from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from PIL import Image
from matplotlib import pyplot
from scipy.spatial.distance import cosine
import os
import pickle
import time
from keras.models import Sequential
from keras.layers import Flatten
from keras import Model
detector = MTCNN()
model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
flatten_output = Sequential()
flatten_output.add(Flatten())
model = Model(input=model.input, output=flatten_output(model.output))

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
    # create the detector, using default weights
    # print('chạy đc')
    # detector = MTCNN()
    # print('chạy ko đc')
    # detect faces in the image
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
    # extract face
    faces = []
    if need_to_extract is False:
        faces = [resize_image(f) for f in filenames]
    else:
        faces = [extract_face(f) for f in filenames]  
    
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=1)
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
        if is_match(query_embedding, embeddings[i][0]) is True:
            retrieval_result.append(embeddings[i][1])
    return retrieval_result

# start_time = time.time()
# curr_dir = os.path.dirname(os.path.realpath(__file__))
# image_dir = os.path.join(curr_dir, 'static/images/faces')
# filenames = []
# onlyFileName = os.listdir(image_dir)

# start4 = time.time()
# for value in absoluteFilePaths(image_dir):
#     filenames.append(value)
# # print(filenames)
# embeddings = get_embeddings(filenames)
# embeded_faces = open('embeded_face.pickle', 'wb')
# pickle.dump(embeddings, embeded_faces)
# embeded_faces.close()
# print('test time 4 ', time.time() - start4)

# start1 = time.time()
# pickle_in = open('embeded_face.pickle', 'rb')
# embeddings = pickle.load(pickle_in)
# pickle_in.close()
# print('test time 1', time.time() - start1)

# pixels = extract_face('viet_huong1.jpg')
# pyplot.imshow(pixels)
# pyplot.show()
# start2 = time.time()
# viet_huong = get_embeddings(['pham_nhat_vuong.jpg'], need_to_extract=True)
# viet_huong = viet_huong[0]
# print('test time 2', time.time() - start2)
# print('shape', sharon_id.shape)
# print('Positive Test')
# for embedding in embeddings:
#     is_match(viet_huong, embedding)
# for i in range(len(embeddings)):
#     print(onlyFileName[i])
#     is_match(viet_huong, embeddings[i])

# print('time: ', time.time() - start_time)
# is_match(embeddings[0], embeddings[1])
# is_match(embeddings[0], embeddings[2])
# print('Negative Tests')
# is_match(embeddings[0], embeddings[3])