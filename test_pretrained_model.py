import os
import cv2
import sys
import numpy as np
from keras import applications
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from train_inception import encode_labels, standardize_data, shuffle_data
from keras.preprocessing.image import ImageDataGenerator


def load_data(dir, max_examples, resize_height, resize_width, label, m_idx):
	""" Loads data from directory dir into a global data list.
		Saves labels into a global labels list. 
		max_examples defines the maximum number of images to import.
		"""
	for idx, i in enumerate(os.listdir(dir)):
		if idx == max_examples: # only read max_examples number of examples
			break
		print("Reading " + label + " " + str(idx))
		img = cv2.imread(dir + "/" + i)
		img = cv2.resize(img, (resize_height, resize_width))
		
		# append data and labels to lists
		# assert(img[:,:,0].all() == img[:,:,1].all() == img[:,:,2].all()) # assert image is grayscale and removing dimension wont affect training
		# img = img[:,:,0] # remove the RGB color dimension (no new data)
		assert(data[m_idx,:,:,:].shape == img.shape) # asserts image matrix is of the intended shape, and fits data matrix
		data[m_idx,:,:,:] = img #add the array to data
		labels.append(label)
		m_idx = m_idx + 1
	
	# assert all labels went into the lists
	assert(len(labels) == m_idx)

	return (data, labels, m_idx)


def load_examples(dir, max_examples, resize_height, resize_width):
	""" Loads un-labeled data from directory dir into a global data list.
		max_examples defines the maximum number of images to import.
		"""
	for idx, i in enumerate(os.listdir(dir)):
		if idx == max_examples: # only read max_examples number of examples
			break
		img = cv2.imread(dir + "/" + i)
		img = cv2.resize(img, (resize_height, resize_width))
		
		# append data to array
		assert(data[idx,:,:,:].shape == img.shape) # asserts image matrix is of the intended shape, and fits data matrix
		data[idx,:,:,:] = img #add the array to data

	return data


def create_model(img_size, num_classes):
	model = applications.resnet50.ResNet50(include_top=False, weights=None, 
											input_shape = (img_size[0], img_size[1], 3), 
											classes=num_classes)
	return model

if __name__ == "__main__":
	# """ The file can run on two modes: 
	#     1. Validation: imports labeled examples, and runs a validation through the model. Prints the accuracy
	#     2. Prediction: imports un-labeled examples, and runs predictions using the model. Prints the predicted label"""
	if sys.argv[1] == "validate":
		curr_m_idx = 0
		data = np.empty((64, 300, 300, 3), dtype=np.uint8)
		labels = []

		data, labels, m_idx = load_data("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/valid/Colonies_Viable_Val", 32, 300,300, "viable", curr_m_idx)
		curr_m_idx = m_idx
		data, labels, m_idx = load_data("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/valid/Diff_And_Bad_Val", 32, 300,300, "unviable", curr_m_idx)
		curr_m_idx = m_idx

		y_val = encode_labels(labels, 'viable', 'unviable')
		x_val = standardize_data(data)

		loaded_model = load_model("/home/nyscf/Documents/sarita/clas/test_colony_inception_mb10_m4000_e30_do1.h5")

		score = loaded_model.evaluate(x_val, y_val, verbose=1)
		print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

	if sys.argv[1] == "predict":
		weights_path = "/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m4135_e100_doNone_preprocessing/chkpt_model.28-acc0.92.hdf5"
		image_path = "/home/nyscf/Documents/test_images"
		img_size = (224,224)


		num_examples_to_load = 10
		data = np.empty((num_examples_to_load, img_size[0], img_size[1], 3), dtype=np.uint8)
		data = load_examples(image_path, num_examples_to_load, img_size[0], img_size[1])
		
		# model = create_model()
		# model.load_weights(weights_path)

		loaded_model = load_model(weights_path)
		predictions = loaded_model.predict(data)
		
		viable_tag = predictions[0]
		correct_pred = 0
		incorrect_pred = 0

		print(predictions.shape)
		print(str(predictions))
		# for pred in predictions:
		# 	if (pred == viable_tag).all():
		# 		correct_pred += 1
		# 	else:
		# 		incorrect_pred += 1

		# print("the number of correct predictions are:", correct_pred)
		# print("the number of incorrect predictions are: ", incorrect_pred)
		# print("the total number of images evaluated are:", len(predictions))

		### TODO: print predicted labels and also probabilities, not binary array of prediction
		

