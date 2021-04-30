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
from datetime import datetime
import pandas


#Importing the pretrained vggface2 model
resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()

#Expression Classification Model
exp_classifier=exp.EmotionRecognition().cuda().eval()
exp_classifier.load_state_dict(torch.load('exp_model/new_mod.pt'))

#Face Recognition Model
clf=pickle.load(open('custom_faces.sav', 'rb'))

#SET THE Transforms
transforms=transforms.Compose([transforms.ToTensor(),transforms.Resize((100,100))])
cap=cv2.VideoCapture('input_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid_fps= int(cap.get(cv2.CAP_PROP_FPS))

ret,k=cap.read()
h,w,_=k.shape
out = cv2.VideoWriter('output.avi',fourcc, cap.get(cv2.CAP_PROP_FPS), (w,h))


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#CSV FILE
output_file=open('result.csv','w')
output_file.write("Date"+","+"User"+","+"Emotion"+"\n")
#User List
fname=open('users.pickle','rb')
user_list=pickle.load(fname)
user_list[-1]="Unknown person"

#Emotions Dictionary
emotion_lab={-1:'Unidentified Emotion',0:'ANGER',1:'JOY',2:'NEUTRAL',3:'SADNESS'}


#Facial feature extraction
detector= dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("exp_model/shape_predictor_68_face_landmarks.dat")
loop_var=0
while True:
	ret,img_orig=cap.read()
	if(ret==False):
		print('VideoCapture not working or done')
		break
	gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
	img=img_orig.copy()
	faces=detector(gray)
	num_users=0
	for face in faces:
		face_features=[]
		x,y,w,h=abs(face.left()),abs(face.top()),abs(face.right())-abs(face.left()),abs(face.bottom())-abs(face.top())
		landmarks = predictor(image=gray,box=face)
		img=img_orig[y:y+h,x:x+w,:]
		gray=gray[y:y+h,x:x+h]
		ex,ey,ew,eh=x,y,w,h
		for n in range(0, 68):
			x = landmarks.part(n).x
			y = landmarks.part(n).y
			face_features.append([x,y])
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = Image.fromarray(np.uint8(img)).convert('RGB')
		img=transforms(img)
		comb_fs=resnet(torch.reshape(img,(1,3,100,100)).cuda())
		face_embed=comb_fs[3].detach().to('cpu').numpy()

		yout=clf.predict_proba(face_embed.reshape(1,512))
		probab=yout[0,np.argmax(yout)]
		res_cls=np.argmax(yout) if probab>=.7 else -1
		user=user_list[res_cls]
		db='User : '+user+'  score:'+str(float(probab))
		if res_cls==-1:
			cv2.rectangle(img_orig, (ex,ey), (ex+ew,ey+eh), (0, 0, 255), 2)
		else:
			cv2.rectangle(img_orig, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)
		if(len(face_features)==0):
			continue
		face_features=torch.tensor(face_features).reshape(1,136).cuda()	
		result=exp_classifier(comb_fs[0],comb_fs[1],comb_fs[2],face_features).cpu()
		result=nn.Softmax()(result).detach().numpy()
		emotion_no=np.argmax(result.ravel())
		emotion=emotion_lab[emotion_no]
		cv2.putText(img_orig, str(emotion), (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
		cv2.putText(img_orig, db,(ex,ey+eh+10),cv2.FONT_HERSHEY_SIMPLEX,.9, (0,255,0), 1, cv2.LINE_AA)
		num_users+=1
		
		
			
		if (loop_var%vid_fps)==0:
			s=str(datetime.now())+","+user+","+emotion+"\n"
			output_file.write(s)
			print(loop_var)
			
	loop_var+=1
	out.write(img_orig)
	

'''
	if len(face_features)==0:
		faces = face_cascade.detectMultiScale(gray,1.7,2)
		if len(faces)==0:
			continue
		for (x, y, w, h) in faces:
			cv2.rectangle(img_orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
			img=img_orig[y:y+h,x:x+w,:]
			gray=gray[y:y+h,x:x+h]
			cv2.imshow('result',img_orig)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		continue



	
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

	cv2.putText(img_orig, db, (50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 1, cv2.LINE_AA)
	#out.write(img_orig)

	cv2.imshow('result',img_orig)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
'''
output_file.close()
out.release()
cap.release()
cv2.destroyAllWindows()