import cv2
import os
import face_recognition
import numpy as np
import pickle




def detect_faces_in_image(file_stream):

  with open('C:\\Users\\WIND\\OneDrive\\Project\\vn_celeb_face_recognition\\FaceRecognitionIdol\\dataset_faces.dat', 'rb') as f:
	  all_face_encodings = pickle.load(f)
  
  known_face_encodings=np.array(list(all_face_encodings.values()))
  known_face_names=list(all_face_encodings.keys())

  img=cv2.imread(file_stream)

  unknown_image=  face_recognition.load_image_file(file_stream)
  face_locations = face_recognition.face_locations(unknown_image)
  unknown_face_encodings = []
  # Get face encodings for any faces in the uploaded image
  unknown_face_encodings = face_recognition.face_encodings(unknown_image,face_locations)
    
  face_names = []    

  for face_encoding in unknown_face_encodings:
        

    match_results = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
    name='unknown'

    face_distances = face_recognition.face_distance(known_face_encodings,face_encoding) 

    best_match_index = np.argmin(face_distances)
    if match_results[best_match_index]:
      name=known_face_names[best_match_index]

    face_names.append(name)

  
  for (top, right, bottom, left), name in zip(face_locations, face_names):
    #Crop face when image full
    img= img[top:bottom,left:right]

    saveToFolder = "/content/drive/My Drive/Project/FACE_IDOL/face_nhan_dien_duoc/"

    cv2.imwrite(saveToFolder+str(name),img)
    print("Crop success!")

  '''   
  if face_names!=[]:
    cv2.imshow('img',img)
    cv2.waitKey()
    print (face_names)
  else:
    print("Khong nhan dien dc")

  '''
  return face_names
def get_faceKnown(folder_image):
  known_names=[]
  known_endcodings=[]

  
  path_train=os.listdir(folder_image)
  
  for i in path_train:
    
    if i.endswith('.png'):
      image_file=os.path.join(folder_image,i)
      imagef = face_recognition.load_image_file(image_file)
      face_location=face_recognition.face_locations(imagef)
      if face_location!=[]:
        unknown_face_encoding=face_recognition.face_encodings(imagef,face_location)[0]
        known_names.append(i)
        known_endcodings.append(unknown_face_encoding)
        
    else:
      print ("FILE TRAIN ERROR")

  all_face_encodings={}
  for i in range(len(known_names)):
    all_face_encodings[known_names[i]]= known_endcodings[i]
  with open('/content/drive/My Drive/PythonCNN/dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
  f.close()
  #return all_face_encodings

    
#detect_faces_in_image('C:\\Users\\WIND\\OneDrive\\Project\\face_Recognitions\\a.jpg')


#get_faceKnown('/content/drive/My Drive/Project/FACE_IDOL/train/')

#sfolder_image="C:\\Users\\WIND\\OneDrive\\Project\\vn_celeb_face_recognition\\test\\000e97227ffa4604b446f7a3235ab877.png"

#detect_faces_in_image(folder_image)