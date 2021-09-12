import collections
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# from CollectDataset import processData
from dataLoader0 import DynamicDataLoader
import model2 as model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape

from tensorflow.keras.layers import Dense, GlobalAveragePooling3D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ConvLSTM2D
#import theano
import matplotlib
import matplotlib.pyplot as plt
import cv2
#from sklearn.model_selection import train_test_split
#from sklearn import model_selection
#from sklearn import preprocessing
from tensorflow.python.client import device_lib

def splitName(name):
    fileName = os.path.splitext(os.path.basename(name))[0]
    res = fileName[fileName.find('_')+1:]
    return (res == '01')


weights_path = './weights/CNN_MLP'

# args
Args = {
    'classes': ['left', 'right'],
    'image_shape': (25, 100, 100, 1),
    'csv_shape': (25, 63),
    'epoch': 20,
    'opt': tf.keras.optimizers.Adam(),
    'loss': tf.keras.losses.SparseCategoricalCrossentropy(),
    'metrics': tf.keras.metrics.SparseCategoricalAccuracy(),
    'callbacks': [tf.keras.callbacks.ModelCheckpoint(os.path.join(weights_path, 'callback_weights'), save_best_only=True, save_weights_only=True)]
}

# data
train_list = list(pathlib.Path('../../dataset/dynamic/train').glob('*/*.jpg'))
print(len(train_list))
val_list = list(pathlib.Path('../../dataset/dynamic/val').glob('*/*.jpg'))
print(len(val_list))

group_train_list = [file for file in train_list if splitName(file)]
print(len(group_train_list))
group_val_list = [file for file in val_list if splitName(file)]
print(len(group_val_list))

# model
mlp = model.MLP_Extract(Args['csv_shape'])
cnn = model.CNN_Extract(Args['image_shape'])
concat = concatenate([mlp.output, cnn.output])
x = Dense(16, activation="relu")(concat)
x = Dense(len(Args['classes']), activation="softmax")(x)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

model.compile(loss=Args['loss'], optimizer=Args['opt'], metrics=[Args['metrics']])
model.summary()

# model.save(os.path.join(weights_path, 'init_weights'))

history = model.fit(
    DynamicDataLoader(Args['classes'], train_list, group_train_list, 32, 25, True, '../../dataset/dynamic/dynamic_gesture_train.csv'),
    validation_data=DynamicDataLoader(Args['classes'], val_list, group_val_list, 32, 25, True, '../../dataset/dynamic/dynamic_gesture_val.csv'),
    epochs=Args['epoch'],
    callbacks=Args['callbacks'])
model.save_weights(os.path.join(weights_path, 'finally_weights'))

# drow acc log
plt.plot(history.history['categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])
plt.show()