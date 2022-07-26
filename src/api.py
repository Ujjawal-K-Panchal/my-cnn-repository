# -*- coding: utf-8 -*-
"""
Title: CNN.

Description: A Convolutional Neural Network.

Created on Sun Jul 10 2022 19:19:01 2022.

@author: Ujjawal .K. Panchal
===

Copyright (c) 2022, Ujjawal Panchal.
All Rights Reserved.
"""
#common libs.
import torch, io, numpy as np
from PIL import Image

#api lib.
from fastapi import FastAPI, File, UploadFile

#import our files.
import main, pipeline
from model import CNN

#some of our assumptions.
OURIMGSIZE = (32, 32) 


#load models.
mnist_model = pipeline.load_model(CNN(1), f"MNIST-{main.MODELNAME}")
fashion_model = pipeline.load_model(CNN(1), f"Fashion-{main.MODELNAME}")

#make api main.
app = FastAPI()

#classnames.
mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fashion_classes = [
					"T-shirt/top",
					"Trouser",
					"Pullover",
					"Dress",
					"Coat",
					"Sandal",
					"Shirt",
					"Sneaker",
					"Bag",
					"Ankle boot",
]



@app.post("/predict/")
async def predict_image_class(
	image_file: UploadFile = File(...),
	dataset_name: str = "MNIST",
) -> str:
	"""
	For a file uploaded by user, accept it and print the show class\
	predicted by model.
	---
	Arguments:
		1. image_file: UploadFile = The file of image which you want to predict for.
		2. dataset_name: str (support: MNIST|Fashion) = Name of the type of dataset it is. 
	"""
	#1. load image file to image.
	bytes_content = await image_file.read()
	image = Image.open(io.BytesIO(bytes_content)).resize(OURIMGSIZE).convert('L')
	np_img = np.array(image)
	np_img = np_img[np.newaxis, np.newaxis, :, :]
	print(f"{np_img.shape}")
	img_tensor = torch.tensor(np_img, dtype = torch.float32)

	#2. predict the output.
	if dataset_name == "MNIST":
		prediction_soft = torch.nn.functional.softmax(mnist_model(img_tensor), dim = 1)
		prediction_hard = prediction_soft.argmax(dim = 1)
		pred_class = mnist_classes[prediction_hard]
	elif dataset_name == "Fashion":
		prediction_soft = torch.nn.functional.softmax(fashion_model(img_tensor), dim = 1)
		prediction_hard = prediction_soft.argmax(dim = 1)
		pred_class = fashion_classes[prediction_hard]
	else:
		raise Exception(f"Dataset '{dataset_name}' is not supported.")
	return {"predicted class": pred_class}