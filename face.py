# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosine
from matplotlib.patches import Rectangle
import pickle


# extract multiple faces from a given group photograph
IMG_SIZE=200
face_coordinates=[]
def extract_faces(pixels):
    face_coordinates.clear()
    face_array_list=[]
    # create the detector, using default weights
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    #print(results)
    # extract the bounding box from the faces
    for i in range(0,len(results)):
        x1, y1, width, height = results[i]['box'] # 0 for first face 1 for 2nd face etc.
        face_coordinates.append([x1,y1,width,height])
        print('face coordinates',[x1,y1,width,height])
        x2, y2 = abs(x1) + width, abs(y1) + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        face_array=cv2.resize(face,(IMG_SIZE,IMG_SIZE))
        face_array_list.append(face_array)
        
    
   
    return face_array_list

# extract faces and calculate face embeddings for a list of photo files
 
def get_embeddings(faces):
    # extract faces
    #faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat

#calculate embedding for known faces
known_faces=[]
known_embedings=[]
known_file =['static/uploads/known/pk.jpg']
#known_file =['static/uploads/known/pp.jpg']
for i in range(0,len(known_file)):
    img_array = plt.imread(known_file[i])
    face_array=extract_faces(img_array)
    known_faces.append(face_array)
    embedding_array=get_embeddings(face_array)
    known_embedings.append(embedding_array)
    


#TEST of Model now
#extract the all faces from group photo and calculate embedding for each face
group_photo='static/uploads/unknown/pk_unkn.jpg'
group_img_array = plt.imread(group_photo)
face_array_list=extract_faces(group_img_array)
group_embedding=get_embeddings(np.array(face_array_list))


#flag=False
score=0.0
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding):
    
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    
    return score

thresh=0.35 #means looking for at least 65% similarity
print("Please note this machine only knows prashant")
group_array=plt.imread(group_photo)
plt.imshow(group_array)
for i in range(0,len(group_embedding)):
    pk_score=is_match(known_embedings[0], group_embedding[i])
    if pk_score < thresh:
        ax=plt.gca()
        x=face_coordinates[i][0]
        y=face_coordinates[i][1]
        width=face_coordinates[i][2]
        height=face_coordinates[i][3]
      
        rect=Rectangle((x,y), width, height,fill=False, color='red')
        ax.add_patch(rect)
        s=str((100-pk_score*100));
        ax.text(x,y,'Prashant - '+s,color='yellow')
     
plt.show()


model = VGGFace()
print(model.summary())


