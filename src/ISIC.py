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
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from keras.layers import Conv2D, UpSampling2D, BatchNormalization,GlobalAveragePooling2D, Dropout
from keras.models import Sequential
import tensorflow_addons as tfa
import os
from keras.layers import Flatten
import tensorflow as tf
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
#print(tf.__version__)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



test_auc = []
test_acc = []
epoch = 50


def import_ISIC(img_dir, label_dir):
    """
    :param img_dir: directory where images are stored
    :param label_dir: directory where labels are stored
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .jpg
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

    # import labels and set image id as index column
    labels = pd.read_csv(label_dir)
    labels = labels.set_index("image")

    tables = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
        img_id = img_path[-16:-4]  # get image id from image path
        extracted_label = labels.loc[img_id]  # extract label from label csv
        if extracted_label[0] == 1:
            extracted_label = 'MEL'
        elif extracted_label[1] == 1:
            extracted_label = 'NV'
        elif extracted_label[2] == 1:
            extracted_label = 'BCC'
        elif extracted_label[3] == 1:
            extracted_label = 'AKIEC'
        elif extracted_label[4] == 1:
            extracted_label = 'BKL'
        elif extracted_label[5] == 1:
            extracted_label = 'DF'
        elif extracted_label[6] == 1:
            extracted_label = 'VASC'
        entry['class'] = extracted_label  

        tables.append(entry)  

    train_labels = pd.concat(tables, ignore_index=True)  
    print(train_labels['class'].value_counts())  
    return train_labels



df = import_ISIC("/home/data_shares/CATSemilie/data/ISIC2018/ISIC2018_Task3_Training_Input", "/home/data_shares/CATSemilie/data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")



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

image_size = 128

def run_model():
	with strategy.scope():
		pretrained_autoencoder = autoencoder(weights='PATH TO PRE-TRAINED WEIGHT', input_shape=(image_size, image_size,3))
		x = pretrained_autoencoder.layers[-13].output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x)
		predictions = Dense(7, activation='softmax')(x)
		

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


		model.fit(train_generator , epochs=30, callbacks = [checkpoint],validation_data=validation_generator, verbose =1 )

		


		print('this is my test score')
		from keras.models import load_model
		new_model = load_model(filepath)
		score = new_model.evaluate(test_generator,return_dict = True, verbose=1)
		print(score)
		test_acc.append(score['accuracy'])
		test_auc.append(score['auc'])
		return score



X_train =df[0:7000]
X_val = df[7000:8000]
X_test = df[8000:]


for i in range(5):

  batch_size = 112
  img_length = 128
  img_width = 128


  train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range = 10,
            horizontal_flip = True)

  val_tes_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,)



  num_classes = len(np.unique(X_train['class']))  



  print(num_classes)

  train_generator = train_data_generator.flow_from_dataframe(dataframe=X_train,
                                                        x_col='path',
                                                        y_col='class',
                                                        target_size=(img_length, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        seed=2)

  validation_generator = val_tes_datagen.flow_from_dataframe(dataframe=X_val,
                                                                 x_col='path',
                                                                 y_col='class',
                                                                 target_size=(img_length, img_width),
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode="categorical",
                                                                 seed=2)

  test_generator = val_tes_datagen.flow_from_dataframe(dataframe=X_test,
                                                           x_col='path',
                                                           y_col='class',
                                                           target_size=(img_length, img_width),
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           class_mode="categorical",
                                                           seed=2)

  run_model()

print(test_acc)
print(test_auc)
avg = sum(test_acc) / len(test_acc)
avg_auc = sum(test_auc) / len(test_auc)
print('average acc is:', avg)  
print('average auc is:', avg_auc)

