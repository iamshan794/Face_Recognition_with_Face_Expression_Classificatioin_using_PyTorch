# Face Recognition and Expression Classificatioin using PyTorch

This is a PyTorch implementation of face recognition and expression classification using face embeddings extracted from InceptionResnetV1 and keypoints extracted from dlib.


# Face Recognition

The InceptionResnetV1 is loaded with pretrained 'VGGFace2' weights. Upon extracting the 512 dimensional face embeddings, a custom large margin classifier is trained to recognize users. In addition, users can be added and removed instantly. 

Steps to add a user:
* Place few samples of the person's face in a labelled folder. This folder has to be created under Userdata.
* Run add_new_user.py
* The face embeddings along with the current user list is updated automatically.

# Expression Classification

A <strong>custom model</strong> which takes in several features from InceptionResnetv1 and keypoints extracted using dlib is trained to classify four fundamental emotions [JOY,SAD,ANGRY and NEUTRAL]. A flowchart of the model is given below.

<img src=https://github.com/iamshan794/Face_Recognition_with_Face_Expression_Classificatioin_using_PyTorch/blob/main/EXP_CLS.jpg width=870 height=940>

# How to run the model?

* Use test_model.py to see the system in action.
* Use detect_vid.py to process a video.

# References:
* Facenet-Pytorch - https://github.com/timesler/facenet-pytorch
* DLIB - https://github.com/davisking/dlib
* VGGFace2 - https://arxiv.org/abs/1710.08092#:~:text=Qiong%20Cao%2C%20Li%20Shen%2C%20Weidi,362.6%20images%20for%20each%20subject.
