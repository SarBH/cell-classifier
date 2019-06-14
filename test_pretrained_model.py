import os
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras import applications
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense




Xval = []
Yval = []


for idx, i in enumerate(os.listdir("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/test/test_folder")):
	if idx == 500:
		break
	print("Reading test image " + str(idx))
	img = cv2.imread("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/test/test_folder/" + i)
	img = cv2.resize(img, (500, 500))
	Xtest.append(img)
    Yval.append("unviable")

loaded_model = load_model("/home/nyscf/Documents/sarita/clas/test_colony_inception_mb10_m500_e30.h5")

score = loaded_model.evaluate(Xtest, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# predictions = model.predict(Xtest, batch_size=None, verbose=0, steps=None, callbacks=None)


"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
"""