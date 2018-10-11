'''
A simple library for processing the image data files.
Author : lianggaoquan

'''
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import to_categorical


def get_names(path):
	return sorted(os.listdir(path))
	
def get_df(path, names):
	df = pd.read_csv(os.path.join(path,names[-1]))
	return df

def get_raw_data(df):
	dic = df.to_dict('index')
	raw_data = []
	for val in dic.values():
		raw_data.append(val)
	return raw_data

def get_precessing_data(path, raw_data):
	data = []
	for e in raw_data:
		temp = {}
		img_path = os.path.join(path,e['file'])
		x = image_transform_to_array(img_path)
		
		temp['array'] = x
		temp['label'] = e['label']
		data.append(temp)
	return data

def get_dataset(data, num_category, train_rate):
	X = []
	y = []
	for i in range(len(data)):
		X.append(data[i]['array'])

	for j in range(len(data)):
		y.append(data[i]['label'])

	X = np.array(X)
	X = X.reshape(X.shape[0],224,224,3)
	y = np.array(y)
	y = y.reshape(y.shape[0],1)
	final_data = np.array(list(zip(X,y)))
	train_size = int(train_rate * final_data.shape[0])
	np.random.shuffle(final_data)

	X_train = []
	for i in range(train_size):
		X_train.append(final_data[:train_size,0][i])
	X_train = np.array(X_train)

	y_train = final_data[:train_size,1]
	y_train = y_train.reshape(y_train.shape[0],1)

	X_test = []
	for j in range(final_data.shape[0] - train_size):
		X_test.append(final_data[train_size:,0][j])

	X_test = np.array(X_test)
	y_test = final_data[train_size:,1]
	y_test = y_test.reshape(y_test.shape[0],1)
	
	y_train = to_categorical(y_train,num_classes=num_category)
	y_test = to_categorical(y_test,num_classes=num_category)
	
	return (X_train, y_train, X_test, y_test)

def process(path, num_category, train_rate = 0.85):
	'''
	merge the methods together
	'''
	names = get_names(path)
	df = get_df(path, names)
	raw_data = get_raw_data(df)
	data = get_precessing_data(path, raw_data)
	dataset = get_dataset(data,num_category,train_rate)
	return dataset

def image_transform_to_array(img_path):
	img = image.load_img(img_path,target_size=(224,224))
	x = image.img_to_array(img)
	return x
	