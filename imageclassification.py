import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

#trainning data generator
traindataGen = ImageDataGenerator(rescale=1./256, zoom_range=0.2, rotation_range = 15, horizontal_flip=True)
directory = 'C:\TaiLieuHocTap\RMIT\A_2023\ML_A23\GroupProject\Flowers'
CLASS_MODE = 'binary'
COLOR_MODE = 'rgb' 
TARGET_SIZE = (256, 256)
BATCH_SIZE = 1 # sau con dung stochastic gradient descent ( xin hon gradient descent)
train_generator = traindataGen.flow_from_directory(
    directory, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=CLASS_MODE, color_mode=COLOR_MODE) 


# create validation data generator
valdataGen = ImageDataGenerator(rescale=1./256)
validation_generator = valdataGen.flow_from_directory(
    directory, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=CLASS_MODE, color_mode=COLOR_MODE)

# create test data generator
testdataGen = ImageDataGenerator(rescale=1./256)
test_generator = testdataGen.flow_from_directory(
    directory, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=CLASS_MODE, color_mode=COLOR_MODE)