import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from facenet_pytorch import InceptionResnetV1
import pickle
import dlib
import cv2
import sklearn
from sklearn import svm

def train_svm(my_embeddings,Y_train):
	X=my_embeddings
	Y=Y_train
	Y=np.array(Y_train)
	clf = svm.SVC(kernel='sigmoid', C=1000,probability=True)
	clf.fit(X, Y)
	print("Trained SVM Model Successfully.")
	filename = 'custom_faces_'+str(int(X.shape[0]/10))+'.sav'
	pickle.dump(clf, open(filename, 'wb'))
	



def get_face(img_orig,detector,predictor):

	
	
	
	gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
	img=img_orig.copy()
	faces=detector(gray)
	for face in faces:
		x,y,w,h=abs(face.left()),abs(face.top()),abs(face.right())-abs(face.left()),abs(face.bottom())-abs(face.top())
		
		img=img_orig[y:y+h,x:x+w,:]
		gray=gray[y:y+h,x:x+h]
		
		while True:
			cv2.imshow("res",img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
	return Image.fromarray(np.uint8(img)).convert('RGB')


def restart_training(resnet):
	#Facial feature extraction
	trans=transforms.Compose([transforms.ToTensor(),transforms.Resize((100,100))])
	detector= dlib.get_frontal_face_detector()
	predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	os.chdir('Userdata')
	count=0
	users_list={}
	fnames=os.listdir()
	fnames=[fnames[i] for i in range(len(fnames)) if os.path.isdir(fnames[i])]
	print("Users found",fnames)
	concatenated=torch.empty(1,3,100,100).cuda()
	Y_train=[]
	for afname in fnames:
		img_names=os.listdir(afname)
		for i in range(len(os.listdir(afname))):
			img=Image.open(os.path.join(afname,img_names[i]))
			img=get_face(np.array(img),detector,predictor)
			if img==None:
				print("skipping images. no face detected for "+afname )
				continue
			img=trans(img)
			img=img.expand(1,3,100,100).cuda()
			concatenated=torch.cat((concatenated,img),0)
			Y_train.append(count)
		users_list[count]=str(afname)	
		count+=1
	comb_fs=resnet(concatenated[1:,:,:,:].cuda())
	my_embeddings=comb_fs[3].detach().to('cpu').numpy()
	#my_embeddings=resnet(concatenated[1:,:,:,:]).detach().to('cpu').numpy()
	
	
	print("Embeddings extraction successful.")
	os.chdir('..')
	np.save('face_embeddings.npy',my_embeddings)
	file_to_write = open("users.pickle", "wb")
	pickle.dump(users_list, file_to_write)
	train_svm(my_embeddings,Y_train)
	
def add_user(name):
	global users_list
	global count
	global embeddings
	os.chdir('Userdata')
	fnames=os.listdir()
	if name not in fnames:
		return "No user to add"
	else:
		
		img_names=os.listdir(name)
		
		for i in range(len(os.listdir(afname))):
			img=Image.open(os.path.join(afname,img_names[i]))
			img=get_face(np.array(img))
			
			img=trans(img)
			img=img.expand(1,3,100,100).cuda()
			concatenated=torch.cat((concatenated,img),0)
		
		users_list[count]=str(afname)	
		count+=1
	comb_fs=resnet(concatenated[1:,:,:,:].cuda())

	#my_embeddings=resnet(concatenated[1:,:,:,:]).detach().to('cpu').numpy()
	
	my_embeddings=comb_fs[3].detatch().to('cpu').numpy()
	os.chdir('..')
	earlier_embeddings=np.load('face_embeddings.npy')
	net_embeddings=np.append(earlier_embeddings,[my_embeddings],axis=0)
	np.save('face_embeddings.npy',net_embeddings)
	return "All Users Added User Successfully."
		
'''
global users_list
global count
global embeddings
resnet = InceptionResnetV1(pretrained='vggface2').cuda()
if os.path.exists('users.pickle'):
	file_to_write = open("users.pickle", "wb")
	users_list=pickle.load(file_to_write)
	embeddings=np.load('face_embeddings.npy')
	current_users=len(embeddings)
	now_users=os.listdir('data')
	if(len(now_users)==current_users):
		print("No new user to add")
	else:
		user_to_add=0
		for user in now_users:
			if user not in user_list.values()
				user_to_add=user
		print(add_users(user_to_add))


else:
	print("User Dictionary not found. Restarting training.")
	restart_training()

os.chdir()
'''
resnet = InceptionResnetV1(pretrained='vggface2').cuda()
print("Restarting training....")
print(restart_training(resnet))
