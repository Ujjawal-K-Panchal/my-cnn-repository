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
import torch
from torch import nn


class CNN(nn.Module):
	"""
	Our CNN Class.
	"""
	def __init__(self, c):
		"""
		Description:
			Just the initiation function.
		Args:
			1. c (int) := number of channels.
		"""
		super().__init__()
		
		#convolutions.
		#x =  32, 32.
		self.conv1 = nn.Conv2d(c, 20, 3, stride = 2) # new_h = ((32 - 3) // s) + 1 = 15 
		self.act1 = nn.ReLU()
		self.mp1 = nn.MaxPool2d(2, 2) #new_h = 15//2 = 7.
		
		self.conv2 = nn.Conv2d(20, 40, 3, stride = 2) # new_h = ((7 - 3) // s) + 1 = 2
		self.act2 = nn.ReLU()
		self.mp2 = nn.MaxPool2d(2, 2) #new_h = 2//2 = 1.
		
		#flatten.
		self.flattener = nn.Flatten()

		#linear.
		self.fc1 = nn.Linear(40, 20) # 40 * 1 * 1.
		self.relu = nn.ReLU()

		self.fc2 = nn.Linear (20, 10)

	def forward(self, x):
		"""
		Description:
			Propogate the input through the network.
		Args:
			1. x  (type: int, shape: batchsize, channels, height, width) := the input batch.
		"""
		x = self.mp1(self.act1(self.conv1(x)))
		x = self.mp2(self.act2(self.conv2(x)))
		x = self.flattener(x)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		return x

if __name__ == "__main__":
	mymodel = CNN(1)
	print(mymodel)