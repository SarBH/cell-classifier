import os
import cv2
import sys
import numpy as np
from keras import applications
from keras.models import Model, load_model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from train_inception import encode_labels, standardize_data, shuffle_data
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels


# def load_examples(dir, max_exampletop_model = Sequential(), resize_height, resize_width):
# 	""" Loads un-labeled data from dtop_model = Sequential()rectory dir into a global data list.
# 		max_examples defines the maxtop_model = Sequential()mum number of images to import.
# 		"""
# 	for idx, i in enumerate(os.listdtop_model = Sequential()r(dir)):
# 		if idx == max_examples: # ontop_model = Sequential()y read max_examples number of examples
# 			break
# 		img = cv2.imread(dir + "/" +top_model = Sequential()i)
# 		img = cv2.resize(img, (resiztop_model = Sequential()_height, resize_width))
		
# 		# append data to array
# 		assert(data[idx,:,:,:].shapetop_model = Sequential()== img.shape) # asserts image matrix is of the intended shape, and fits data matrix
# 		data[idx,:,:,:] = img #add ttop_model = Sequential()e array to data

# 	return data



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
	""" The file can run on two modes: 
	    1. Validation: imports labeled examples, and runs a validation through the model. 
						Prints the accuracy. Creates a confusion matrix
	    2. Prediction: imports un-labeled examples, and runs predictions using the model. 
						Prints the predicted label
	"""

	model_path = "/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m4135_e100_do0.4/chkpt_model.32-acc0.82.hdf5"
	test_data_dir = "/home/nyscf/Documents/test_images/"
	img_size = (224,224)
	num_examples_to_load = 22
	valid_batch_size = 1

	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(test_data_dir,
													target_size=img_size,
													batch_size=valid_batch_size,
													class_mode='binary', 
													shuffle = False)
	print("datagen created")

	base_model = applications.resnet50.ResNet50(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=2)
	
	top_model = Sequential()
	top_model.add(Dense(1, activation='sigmoid'))
	output_layer = base_model.output
	output_layer = GlobalAveragePooling2D()(output_layer)
	# top_model.add(output_layer)

	model = Model(inputs=base_model.input, outputs=top_model(output_layer))

	model.load_weights(model_path)

	model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ["acc"]) #used accuracy metric for 2 classes, supposedly catetgorical acc gets selected automatically when using cat cross ent

	prob = model.predict_generator(test_generator, steps=num_examples_to_load, verbose=1)
	print(prob)
	
	

	y_pred = (prob < 0.5).astype(np.int)
	# y_pred = np.argmax(prob, axis=1)
	print("y_pred: ", y_pred)
	
	y_true = test_generator.classes
	print("y_true: ", y_true)

	
	target_names = ['bad','viable']


	print(classification_report(y_true, y_pred, target_names=target_names))
	cm = confusion_matrix(y_true, y_pred)
	print(cm)


	# Plot non-normalized confusion matrix
	plot_confusion_matrix(cm, classes=target_names,
						title='Confusion matrix, without normalization')

	plt.show()
		

"""

	if sys.argv[1] == "predict":
		model_path = "/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m4135_e100_doNone_preprocessing/chkpt_model.28-acc0.92.hdf5"
		image_path = "/home/nyscf/Documents/test_images/bad"
		img_size = (224,224)
		num_examples_to_load = 13

		image_gen = ImageDataGenerator()
		gen = image_gen.flow_from_directory(image_path, batch_size=1)

		index=0
		image, label = gen._get_batches_of_transformed_samples(np.array([index]))
		image_name = gen.filenames[index]

		


		data = np.empty((num_examples_to_load, img_size[0], img_size[1], 3), dtype=np.uint8)
		data = load_examples(image_path, num_examples_to_load, img_size[0], img_size[1])
		
		# model = create_model()
		# model.load_weights(model_path)

		loaded_model = load_model(model_path)
		predictions = loaded_model.predict(data)
		
		viable_tag = predictions[0]
		correct_pred = 0
		incorrect_pred = 0

		print(predictions.shape)
		print(str(predictions))

		prediction_csv = pd.DataFrame(predictions, columns=['predictions']).to_csv('/home/nyscf/Documents/test_images/predictions.csv')

"""