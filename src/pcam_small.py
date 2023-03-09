from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
import h5py
from matplotlib import pyplot as plt
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from keras.layers import Conv2D, UpSampling2D, BatchNormalization,GlobalAveragePooling2D, Dropout
from keras.models import Sequential
import tensorflow_addons as tfa
import os
from keras.layers import Flatten
import tensorflow.keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from numpy import load
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator





print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.python.ops.numpy_ops.np_config as np_config
gpus = tf.config.experimental.list_physical_devices('GPU')
devices = tf.config.list_physical_devices()
print(devices)
print('hello I am starting')
print(tf.__version__)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

 
batch_size = 32
image_size = 96


test_acc = []
test_auc = []
epoch =1

# Define autoencoder architecture for dimension change. 
def autoencoder(weights=None, input_shape=None):
    input_shape = _obtain_input_shape(input_shape,
                                    default_size=224,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    require_flatten=False,
                                    weights=weights)
    img_input = Input(shape=input_shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    


    
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


    inputs = img_input
    model = Model(inputs, decoded, name='autoencoder_today')
    if weights is not None:
        model.load_weights(weights)

    return model



def run_model():
	with strategy.scope():
		pretrained_autoencoder = autoencoder(weights='PATH TO WEIGHT FILE', input_shape=(image_size, image_size,3))
		x = pretrained_autoencoder.layers[-13].output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x)
		redictions = Dense(2048, activation='relu')(x)
		predictions = Dense(1, activation='sigmoid')(x)
		

		model = Model(inputs=pretrained_autoencoder.input, outputs=predictions)
		print(model.summary())
		print('batch size is:',batch_size)
		print('number of epochs:',epoch)


		########## COMPILE #########
	
		optimizer = keras.optimizers.Adam(amsgrad = True)
		learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.3, min_lr=0.000001)
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights= True)
		
		filepath="weights.best.autoencoder.hdf5"
		checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
		

		

		model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name = 'auc'),'accuracy'])

		hist = model.fit(
        		train_generator,
			validation_data = (validation_generator),
        		epochs=30,
 			batch_size=batch_size,
			callbacks=[checkpoint],                 
			verbose = 1)

		for key in hist.history:
			print(key)
		print('this is my test score')
		
		new_model = load_model(filepath)
		score = new_model.evaluate(test_generator,return_dict = True, verbose=1)
		print(score)
		test_acc.append(score['accuracy'])
		test_auc.append(score['auc'])
		return score

for i in range(5):

    datagen = ImageDataGenerator(
      rescale = 1/255,
      rotation_range = 15,
      brightness_range = [0.7, 1.1],
      shear_range=5,
      zoom_range = 0.2)




    train_generator = datagen.flow_from_directory(
      '/home/data_shares/purrlab/PCam/emilie_pcam/data/train',
      class_mode='binary',
      target_size=(image_size, image_size),
      shuffle = True,
      seed=43,
      batch_size=batch_size)


    datagen_u = ImageDataGenerator(rescale=1/255)

    validation_generator = datagen_u.flow_from_directory(
      '/home/data_shares/purrlab/PCam/emilie_pcam/data/validation',
      class_mode='binary',
      target_size=(image_size, image_size),
      shuffle=False,
      seed=43,
      batch_size=batch_size)

    test_generator = datagen_u.flow_from_directory(
      '/home/data_shares/purrlab/PCam/emilie_pcam/data/test',
      class_mode='binary',
      target_size=(image_size, image_size),
      shuffle=False,
      seed=43,
      batch_size=batch_size)
    run_model()

print(test_acc)
print(test_auc)
avg = sum(test_acc) / len(test_acc)
avg_auc = sum(test_auc) / len(test_auc)
print('average acc is:', avg)
print('average auc is:', avg_auc)
