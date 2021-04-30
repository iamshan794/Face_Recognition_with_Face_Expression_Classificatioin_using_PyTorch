import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import torchvision.utils
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
import os
from facenet_pytorch import InceptionResnetV1
import exp_model.exp_classifier_best as exp
import sklearn
from sklearn import svm
import pickle
import cv2
import dlib

#Importing the pretrained vggface2 model

resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()

exp_classifier=exp.EmotionRecognition().cuda().eval()
exp_classifier.load_state_dict(torch.load('exp_model/new_mod.pt'))

#SET THE transform
transforms=transforms.Compose([transforms.ToTensor(),transforms.Resize((100,100))])
cap=cv2.VideoCapture('input_video.mp4')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#emotion_lab={-1:'Unidentified Emotion',0:'ANGER',1:'FEAR',2:'JOY',3:'NEUTRAL',4:'SADNESS'}
emotion_lab={-1:'Unidentified Emotion',0:'ANGER',1:'JOY',2:'NEUTRAL',3:'SADNESS'}

#Facial feature extraction
detector= dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("exp_model/shape_predictor_68_face_landmarks.dat")
ret,img_orig=cap.read()
h,w,c=img_orig.shape
#VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, cap.get(cv2.CAP_PROP_FPS), (w,h))

while True:
	ret,img_orig=cap.read()
	if(ret==0):
		break
	gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
	img=img_orig.copy()
	faces=detector(gray)
	face_features=[]
	for face in faces:
		x,y,w,h=int(face.left()),int(face.top()),int(face.right())-int(face.left()),int(face.bottom())-int(face.top())
		cv2.rectangle(img_orig, (x,y), (x+w,y+h), (255, 0, 0), 2)
		landmarks = predictor(image=gray,box=face)
		img=img_orig[y:y+h,x:x+w,:]
		gray=gray[y:y+h,x:x+h]
		for n in range(0, 68):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			face_features.append([x,y])
	if len(face_features)==0:
		faces = face_cascade.detectMultiScale(gray,1.7,2)
		if len(faces)==0:
			continue
		for (x, y, w, h) in faces:
			cv2.rectangle(img_orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
			img=img_orig[y:y+h,x:x+w,:]
			gray=gray[y:y+h,x:x+h]
			out.write(img_orig)
		continue
		
	ff=torch.tensor(face_features)
	try:
		face_features=ff.reshape(1,136).cuda()	
	except:
		continue

	if(ret==False):
		print('VideoCapture not working or done')
		break
	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(np.uint8(img)).convert('RGB')
	img=transforms(img)
	
	comb_fs=resnet(torch.reshape(img,(1,3,100,100)).cuda())
	result=exp_classifier(comb_fs[0],comb_fs[1],comb_fs[2],face_features).cpu()
	result=nn.Softmax()(result).detach().numpy()
	print(result)
	
	#probab=yout[0,np.argmax(yout)]
	#res_cls=np.argmax(yout) if yout[0,np.argmax(yout)]>=.7 else -1
	#user=cls_lab[res_cls]
	emotion_no=np.argmax(result.ravel())
	emotion=emotion_lab[emotion_no]
	db='  Emotion : '+str(emotion)

	cv2.putText(img_orig, db, (50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 1, cv2.LINE_AA)
	#out.write(img_orig)

	out.write(img_orig)
out.release()
cap.release()
cv2.destroyAllWindows()