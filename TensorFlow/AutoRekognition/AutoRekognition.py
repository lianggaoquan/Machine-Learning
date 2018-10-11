'''
AutoRekognition API
Author : lianggaoquan

'''

from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import *
from image_process import process, image_transform_to_array

def get_base_model(base_path):
	base_model = load_model(base_path)
	return base_model

def rebuild(base_model, num_category):
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024,activation='relu')(x)
	pred = Dense(num_category,activation='softmax')(x)
	
	model = Model(inputs=base_model.input,outputs=pred)
	return model

def fix_layer(model):
	for layer in model.layers[:175]:
		layer.trainable = False

	for layer in model.layers[175:]:
		layer.trainable = True

	return model

def fit(model, X_train, y_train, epochs, batch_size):
	return model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size)

def train(base_path, imgs_path, num_category, epochs=2, batch_size=4):
	base_model = get_base_model(base_path)
	model = rebuild(base_model,num_category)
	model = fix_layer(model)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
	(X_train,y_train,X_test,y_test) = process(imgs_path,num_category)
	print('before training...')
	print(test(model,X_test,y_test))
	print('\n===========training=============\n')
	hist = fit(model,X_train,y_train,epochs,batch_size)
	return model, hist, (X_test,y_test)

def test(model,X_test,y_test):
	return model.evaluate(X_test,y_test)

def predict(model,img_path):
	img_array = image_transform_to_array(img_path)
	img_array = img_array.reshape(1,224,224,3)
	return model.predict(img_array)

def AutoRekognition(base_path, imgs_path, num_category, _predict='N', predict_path=None):
	model,hist,(X_test,y_test) = train(base_path, imgs_path, num_category)
	print('evaluating...')
	print(test(model,X_test,y_test))
	if _predict == 'Y':
		print(predict(model,predict_path))
	return model