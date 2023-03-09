from keras.preprocessing.image import ImageDataGenerator
import random
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from time import time
import copy
import numpy as np
import tensorflow_addons as tfa
from keras.layers import Input
from autoencoder_utils import *
import math
import h5py
import keras
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
import tensorflow as tf
import pandas as pd

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
devices = tf.config.list_physical_devices()
print(devices)
print(tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():

	batch_size = 128
	image_size = 224
	print(batch_size)



	#### READ IN DATA ####
	datagen = ImageDataGenerator(rescale=1./255)

	train_ds = datagen.flow_from_directory(
		directory = '/home/common/datasets/imagenet2012/train',
		target_size=(image_size, image_size),
		color_mode='rgb',
		batch_size=batch_size,
		shuffle = False,
		subset='validation',
		class_mode='input')

	datagen = ImageDataGenerator(rescale=1./255)

	val_ds = datagen.flow_from_directory(
		directory= '/home/common/datasets/imagenet2012/val',
		target_size=(image_size, image_size),
		color_mode='rgb',
		batch_size=batch_size,
		shuffle=False,
		subset='validation',
		class_mode='input')


	
	autoencoder = Sequential()
	img_input  = Input(shape=(224,224,3))


	x=Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(img_input)
	x=Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=MaxPooling2D((2, 2))(x)
	x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=MaxPooling2D((2, 2))(x)
	x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=MaxPooling2D((2, 2))(x)
	x=Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)	
	x=UpSampling2D((2, 2))(x)
	x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=UpSampling2D((2, 2))(x)
	x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=UpSampling2D((2, 2))(x)
	x=Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	x=Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
	decoded=Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


	autoencoder = Model(img_input, decoded)	
	optimizer = keras.optimizers.Adam(lr=0.001)
	autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy','mse'])

	print(autoencoder.summary())

	epochs = 200
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=30, start_from_epoch=50)
	print('i am running')
	filepath="model-runs/imagenet_trail_1.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	tensorboard = TensorBoard(log_dir="newest_logs/{}".format(time()))

train_steps =  len(train_ds.labels)/ batch_size
val_steps = len(val_ds.labels) / batch_size

history = autoencoder.fit_generator(train_ds,
	epochs=epochs,
	steps_per_epoch=train_steps, 
	shuffle=True,validation_data=(val_ds),
	validation_steps=val_steps, 
	callbacks=[checkpoint], 
	verbose = 1)


autoencoder.save('models/_test_AE_imagenet.h5')



train_loss = list(history.history['loss'])
train_acc = list(history.history['accuracy'])
val_loss = list(history.history['val_loss'])
val_acc = list(history.history['val_accuracy'])
mse = list(history.history['mse'])
val_mse = list(history.history['val_mse']) 
df_loss = pd.DataFrame({'train_loss':train_loss, 'train_acc':train_acc, 'val_loss':val_loss, 'val_acc':val_acc, 'mse':mse,'val_mse':val_mse})

df_loss.to_csv("model-runs/callback__AE_ImageNet.csv", index=False)
