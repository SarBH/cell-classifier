from keras.models import load_model
import os
import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from keras import applications, optimizers
from datetime import datetime, timedelta
import csv
# import function that extracts attributes from the name of the file
# from labeler.split_by_well_edit import extract_attributes


def extract_attributes(image_name):

    # Extract attributes from name
    attr_list = image_name.split("_")
    run_id = attr_list[0]
    plate_id = attr_list[1] + "_" + attr_list[2]
    well_id = attr_list[6]
    date_taken = datetime.strptime(attr_list[3], '%m-%d-%Y-%I-%M-%S-%p')

    return (run_id, plate_id, well_id, date_taken)



model_path = "/home/nyscf/Documents/sarita/cell-classifier/inception/chkpt_model.36-.hdf5"

img_size = (300,300)
num_examples_to_load = 22
valid_batch_size = 1
num_classes = 2


# FROM CHECKPOINT:
# base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
#                                                 input_shape = (img_size[0], img_size[1], 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# predictions = Dense(num_classes, activation = 'softmax')(x)
	
# model = Model(inputs=base_model.input, outputs=predictions)

# model.load_weights(model_path)

# model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ["acc"]) #used accuracy metric for 2 classes, supposedly catetgorical acc gets selected automatically when using cat cross ent

# model.save("/home/nyscf/Documents/sarita/cell-classifier/inception/model_inception_m6662_e36_acc96_06_02.h5")




#FROM H5 MODEL:
model = load_model("/home/nyscf/Documents/sarita/models/inception/model_inception_m6662_e36_acc96_06_02.h5")
counter = 0

for i in os.listdir("/home/nyscf/Desktop/Classification_Model/data/test/NotSures/1Good"):
    image = cv2.imread("/home/nyscf/Desktop/Classification_Model/data/test/NotSures/1Good/" + i)
    out = image.copy()
    image = cv2.resize(image, img_size)
    # print(image)
    image = image.astype("float") / 255.0
    # print(image)
    images = [image]
    images = np.array(images)


    run_id, plate_id, well_id, date_taken = extract_attributes(i)

    prediction = model.predict(images, verbose=1, batch_size=1)
    print("Prediction for ", i, "= " , str(prediction))

    if prediction[0][0] < 0.5:
        # c = "bad "+str(np.around(prediction[0][1], decimals=2))
        # color = (255, 0, 0)
        # cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imwrite("/home/nyscf/Desktop/Classification_Model/data/test/second100_bad/" + i + ".jpg", out)
        class_predicted = 'bad'
        certainty = str(np.around(prediction[0][1], decimals=2))
        
    else:
        # color = (0, 255, 0)
        # c = "good "+str(np.around(prediction[0][0], decimals=2))
        # cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imwrite("/home/nyscf/Desktop/Classification_Model/data/test/second100_good/" + i + ".jpg", out)
        class_predicted = 'good'
        certainty = str(np.around(prediction[0][0], decimals=2))

    row_in_csv = [counter, run_id, plate_id, well_id, date_taken, class_predicted, certainty]

    with open('/home/nyscf/Desktop/Classification_Model/data/test/NotSures/predictions.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row_in_csv)


    # cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # cv2.imwrite("/home/nyscf/Desktop/labeled_colony_by_model/day9/all_labels/" + i + "__" + str(counter) + ".jpg", out)
    counter += 1
csvFile.close()
