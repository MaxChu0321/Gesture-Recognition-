import os
import cv2
import pathlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class DynamicDataLoader(tf.keras.utils.Sequence):
  def test3():
    patch_size = 25
    path = './dynamic_gesture_train.csv'
    df = pd.read_csv(path)
    Z_tr = []
    dfClass = df[df['class']=='left']

    Z_tr = np.expand_dims(np.array(dfClass[0:patch_size]), 0)

    for i in range(patch_size, dfClass.iloc[-1].name+1, patch_size):
        print([i, i+patch_size])
        Z_tr = np.append(Z_tr, np.expand_dims(np.array(dfClass[i:i+patch_size]), 0), axis=0)
    
    Z_tr = np.delete(Z_tr, [0, 1], 2).astype(np.float32)
    print(Z_tr)
    print(Z_tr.shape)

def test4():
    X_tr=[]          # variable to store entire dataset
    Y_tr=[]
  
    class_list = ['left', 'right']
    
    img_rows,img_cols=64,64

    from tqdm import tqdm
    csv_path = os.path.join("../dynamic_gesture.csv")
    df = pd.read_csv(csv_path)
    ls_path = os.path.join("../rgb/")

    listing = os.listdir(ls_path)
    #csv_listing = os.listdir(csv_path)

    frames = []
    img_label = []
    img_depth=1
    for imgs in listing:
        n = imgs[:imgs.find('_')]
        img = os.path.join(ls_path, imgs)
        frame = cv2.imread(img)
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_label.append(class_list[0])
        if int(n) != img_depth:
            #print('change')
            img_depth=img_depth+1
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)
            frames = []
            Y_tr.append(img_label)
            img_label=[]
      
        frames.append(gray)

    input_img = np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    X_tr.append(ipt)
    print(ipt.shape)
    print(len(X_tr))


# class TestDataLoader(tf.keras.utils.Sequence):
#   def __init__(self, img, size, lmList=[]):
#     self.img = img
#     self.size = size
#     self.lmList = lmList
  
#   def __len__(self):
#     return 1

#   def __getitem__(self, idx):
#     result_img = np.expand_dims(self.get_image(), axis=0)

#     if len(self.lmList) > 0:
#       result_csv = np.expand_dims(self.get_csv(), axis=0)
#       return [result_csv, result_img]
      
#     return result_img
  
#   def get_image(self):
#     img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (self.size, self.size))
#     img = np.expand_dims((np.array(img) / 255).astype(np.float), axis=2)
#     return img

#   def get_csv(self):
#     csv = np.array([self.lmList[0][1], self.lmList[0][2], self.lmList[0][3]])
#     csv = np.expand_dims(csv, axis=0)

#     for points in self.lmList[1:]:
#         csv = np.append(csv, [points[1], points[2], points[3]])
#     csv = csv.astype(np.float32)
#     return csv


#   patch_size=25
#   df = pd.read_csv(csv_listing)
#   result_csv = []
#   # dfClass = df[df['class']==pathlib.PurePath(listing[0]).parent.name]
#   dfClass = df[df['class']=='left']
#   result_csv = dfClass[dfClass['image']==os.path.splitext(pathlib.PurePath(listing[0]).name)[0]]
#   Z_tr = np.expand_dims(np.array(dfClass[0:patch_size]), 0)

#   for i in range(patch_size, dfClass.iloc[-1].name+1, patch_size):
#     Z_tr = np.append(Z_tr, np.expand_dims(np.array(dfClass[i:i+patch_size]), 0), axis=0)

#     print(Z_tr)
#     print(Z_tr.shape)

#   dfClass[0:patch_size]
#   # for csvs in listing[1:]:
#     # n = csvs[:csvs.find('_')]
#     # csv = os.path.join(csv_path, csvs)
#     # csvs = pathlib.PurePath(csvs)
#     # class_name = csvs.parent.name
#     # file_name = os.path.splitext(csvs.name)[0]
#     # dfClass = df[df['class']==class_name]
#     # result_csv = np.append(result_csv, dfClass[dfClass['image']==int(file_name)], axis=0)
#     # Z_tr.append(result_csv)
      
#   result_csv = np.delete(result_csv, [0, 1], 1).astype(np.float32)

