import os
import cv2
import pathlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf

# 靜態影像
class DataLoader(tf.keras.utils.Sequence):
  def __init__(self, class_list, img_path: str, batch_size: int, csv: bool=False, csv_path: str=''):
    self.class_list = class_list
    self.img_path = img_path
    self.batch_size = batch_size
    self.csv = csv
    self.csv_path = csv_path
  
  def __len__(self):
    return np.ceil(len(self.img_path) / self.batch_size).astype(np.int)

  def __getitem__(self, idx):
    batch_path = self.img_path[idx*self.batch_size: (idx+1)*self.batch_size]
    batch_img, batch_label = self.get_image(batch_path)

    if self.csv:
        batch_csv = self.get_csv(batch_path)
        return [batch_csv, batch_img], batch_label

    return batch_img, batch_label
  
  def on_eopch_end(self):
    random.shuffle(self.img_path)
  
  def get_image(self, path):
    img_list = []
    img_label = []
    for p in path:
      img_list.append(plt.imread(p))
      img_label.append(self.class_list.index(p.parts[-2]))

    result_img = np.expand_dims((np.array(img_list) / 255).astype(np.float), axis=3)
    result_label = np.array(img_label).astype(np.float32)
    return result_img, result_label

  def get_csv(self, path):
      df = pd.read_csv(self.csv_path)
      result_csv = []
      dfClass = df[df['class']==pathlib.PurePath(path[0]).parent.name]
      result_csv = dfClass[dfClass['image']==int(os.path.splitext(pathlib.PurePath(path[0]).name)[0])]

      for p in path[1:]:
          p = pathlib.PurePath(p)
          class_name = p.parent.name
          file_name = os.path.splitext(p.name)[0]
          dfClass = df[df['class']==class_name]
          result_csv = np.append(result_csv, dfClass[dfClass['image']==int(file_name)], axis=0)
      
      result_csv = np.delete(result_csv, [0, 1], 1).astype(np.float32)
      return result_csv

# 靜態影像
class TestDataLoader(tf.keras.utils.Sequence):
  def __init__(self, img, size, lmList=[]):
    self.img = img
    self.size = size
    self.lmList = lmList
  
  def __len__(self):
    return 1

  def __getitem__(self, idx):
    result_img = np.expand_dims(self.get_image(), axis=0)

    if len(self.lmList) > 0:
      result_csv = np.expand_dims(self.get_csv(), axis=0)
      return [result_csv, result_img]
      
    return result_img
  
  def get_image(self):
    img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (self.size, self.size))
    img = np.expand_dims((np.array(img) / 255).astype(np.float), axis=2)
    return img

  def get_csv(self):
    csv = np.array([self.lmList[0][1], self.lmList[0][2], self.lmList[0][3]])
    csv = np.expand_dims(csv, axis=0)

    for points in self.lmList[1:]:
        csv = np.append(csv, [points[1], points[2], points[3]])
    csv = csv.astype(np.float32)
    return csv

# 動態影像
class DynamicDataLoader(tf.keras.utils.Sequence):
  '''
  param:
    class_list - 類別名稱list
    src_path - 完整的影像路徑list(包含所有影像). ex: ['./<class_name>/01_01.jpg', './<class_name>/01_02.jpg', ..., './<class_name>/05_02.jpg']
    img_path - 每組連續影像的第一張影像list. ex: ['./<class_name>/01_01.jpg', './<class_name>/02_01.jpg', ..., './<class_name>/05_01.jpg']
    batch_size - batch size
    patch_size - 一組連續影像的影像數量. ex: patch_size=2, 01_01.jpg ~ 01_02.jpg
    csv - 是否也要處理csv檔
    csv_path - csv檔路徑
  '''
  def __init__(self, class_list, src_path: str, img_path: str, patch_size: int, batch_size: int, csv: bool=False, csv_path: str=''):
    self.class_list = class_list
    self.src_path = src_path
    self.img_path = img_path
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.csv = csv
    self.csv_path = csv_path
  
  def __len__(self):
    return np.ceil(len(self.img_path) / self.batch_size).astype(np.int)

  def __getitem__(self, idx):
    batch_path = self.img_path[idx*self.batch_size: (idx+1)*self.batch_size]
    batch_img, batch_label = self.get_image(batch_path)

    if self.csv:
        batch_csv = self.get_csv(batch_path)
        return [batch_csv, batch_img], batch_label

    return batch_img, batch_label
  
  def on_eopch_end(self):
    random.shuffle(self.img_path)
  
  '''
  param:
    path - 一個batch的圖片路徑
  '''
  def get_image(self, path):
    img_list = []
    img_label = []
    temp_list = []

    for p in path:
      img_index = self.src_path.index(p)  # 找出此張影像在原list的index
      patch_set_path = self.src_path[img_index:(img_index+self.patch_size)]
      img_label.append(self.class_list.index(p.parts[-2]))

      for patch_path in patch_set_path:
        temp_list.append(plt.imread(patch_path))

      img_list.append(temp_list)
      temp_list = []

    result_img = np.expand_dims((np.array(img_list) / 255).astype(np.float), axis=4)
    result_label = np.array(img_label).astype(np.float32)
    return result_img, result_label

  '''
  param:
    path - 一個batch的圖片路徑
  '''
  def get_csv(self, path):
      df = pd.read_csv(self.csv_path)
      dfClass = df[df['class']==pathlib.PurePath(path[0]).parent.name]
      dfIndex = dfClass[dfClass['image']==os.path.splitext(pathlib.PurePath(path[0]).name)[0]].index
      result_csv = np.expand_dims(df.loc[dfIndex[0]:(dfIndex[0]+self.patch_size-1)], axis=0)

      for p in path[1:]:
          p = pathlib.PurePath(p)
          class_name = p.parent.name
          file_name = os.path.splitext(p.name)[0]
          dfClass = df[df['class']==class_name]
          dfIndex = dfClass[dfClass['image']==file_name].index
          result_csv = np.append(result_csv, np.expand_dims(df.loc[dfIndex[0]:(dfIndex[0]+self.patch_size-1)], axis=0), axis=0)

      result_csv = np.delete(result_csv, [0, 1], 2).astype(np.float32)
      return result_csv
