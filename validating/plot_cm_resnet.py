import os
import cv2
import sys
import numpy as np
from keras import applications
from keras.models import Model, load_model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling3D, add
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report




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

	model_path = "/home/nyscf/Documents/sarita/models/resnet/2019-07-11/chkpt_model.32-acc0.95.hdf5"
	validation_data_dir = "/home/nyscf/Desktop/Classification_Model/data/validation_from_datagen/batch2"
	img_size = (400,400)
	num_examples_to_load = 892
	valid_batch_size = 1
	num_classes = 2

	validation_datagen = ImageDataGenerator(rescale=1./255)
	validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
													target_size=(img_size[0], img_size[1]),
													batch_size=valid_batch_size,
													class_mode=None, 
													shuffle = False)
	print("datagen created")

	# base_model = applications.resnet50.ResNet50(include_top=False, weights=None, 
    #                                             input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
	# base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
    #                                             input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
	# base_model.add(GlobalAveragePooling2D())
	# base_model.add(Dense(num_classes, activation = 'softmax'))
	
	# x = base_model.output
	# x = GlobalAveragePooling2D()(x)
	# # x = Dense(1024, activation = 'relu')
	# predictions = Dense(num_classes, activation = 'softmax')(x)
	# model = Model(inputs=base_model.input, outputs=base_model.output)

	model = load_model("/home/nyscf/Documents/sarita/models/resnet_builder/2019-07-12/chkpt_model.11-acc0.91.hdf5")
	# model = Sequential()
	# model.add(prev_model)
	# # model.add(Dense(256, activation = 'relu'))
	# # model.add(Dropout(0.5))
	# model.add(Dense(num_classes, activation = 'softmax'))

	# top_model = Sequential()
	# top_model.add(Dense(num_classes, activation='softmax')) #sigmoid for resnet
	# output_layer = base_model.output
	# output_layer = GlobalAveragePooling2D()(output_layer)
	# top_model.add(output_layer)
	# model = Model(inputs=base_model.input, outputs=top_model(output_layer))

	# model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"]) 
	#used accuracy metric for 2 classes, supposedly catetgorical acc gets selected automatically when using cat cross ent


	# model.load_weights(model_path)

	
	# model = load_model("/home/nyscf/Documents/sarita/cell-classifier/inception/model_inception_m6662_e18_acc93_06_02.h5")
	
	# Predict and store predicted labels in y_pred
	prob = model.predict_generator(validation_generator, steps=num_examples_to_load, verbose=1)
	print(prob)
	y_pred = np.argmax(prob, axis=1)
	print("y_pred: ", y_pred)
	print("the number of examples predicted are:", y_pred.shape)
	# Collect y_true
	y_true = validation_generator.classes
	print("y_true: ", y_true)

	if num_classes == 2:
		target_names = ['bad', 'good']
	if num_classes == 3:
		target_names = ['dead','diff', 'viable']
	print(classification_report(y_true, y_pred, target_names=target_names))
	cm = confusion_matrix(y_true, y_pred)
	print(cm)


	# Plot non-normalized confusion matrix
	plot_confusion_matrix(cm, classes=target_names,
						title='Confusion matrix, without normalization')

	plt.show()
		
