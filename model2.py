import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Conv2D, MaxPooling2D, Conv3D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.backend import batch_normalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D,Conv2D,AveragePooling2D,AveragePooling3D
from tensorflow.keras.layers import ConvLSTM2D
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D,GlobalAveragePooling2D

img_rows,img_cols=64,64
patch_size = 25    # img_depth or number of frames used for each video
weight_decay = 0.00005
nb_classes = 2
l2=keras.regularizers.l2
def CNN_Extract(input_shape):
    model = Sequential()
    model.add(Conv3D(16,(3,3,3),
                            #input_shape=(patch_size, img_cols, img_rows, 3),
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Conv3D(16,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2a_a', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))


    model.add(Conv3D(32,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2b_a', activation = 'relu'))
    model.add(Conv3D(32,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2b_b', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2,2)))


    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_a', activation = 'relu'))
    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_b', activation = 'relu'))
    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_c', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2,2)))


    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_a', activation = 'relu'))
    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_b', activation = 'relu'))
    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_c', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))






    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                    strides=(1,1),padding='same',
                        kernel_initializer='he_normal', recurrent_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                        return_sequences=True, name='gatedclstm2d_2'))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                    strides=(1,1),padding='same',
                        kernel_initializer='he_normal', recurrent_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                        return_sequences=True, name='gatedclstm2d_3'))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                    strides=(1,1),padding='same',
                        kernel_initializer='he_normal', recurrent_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                        return_sequences=True, name='gatedclstm2d_4'))


    #model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
    #model.add(Flatten())
    model.add(GlobalAveragePooling3D())
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    return model

def MLP_Extract(input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(LSTM(16, return_sequences=True))
    model.add(Activation('relu'))
    model.add(LSTM(16, return_sequences=True))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    return model


    #第15,97行input_shape