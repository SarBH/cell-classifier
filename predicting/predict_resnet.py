import os
import cv2
import sys
import numpy as np
from keras import applications
from keras.models import Model, load_model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling3D, add
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from datetime import datetime, timedelta
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def extract_attributes(image_name):

    # Extract attributes from name
    attr_list = image_name.split("_")
    run_id = attr_list[0]
    plate_id = attr_list[1] + "_" + attr_list[2]
    well_id = attr_list[6]
    date_taken = datetime.strptime(attr_list[3], '%m-%d-%Y-%I-%M-%S-%p')

    return (run_id, plate_id, well_id, date_taken)







if __name__ == "__main__":
    model_path = "/home/nyscf/Documents/sarita/models/resnet_builder/2019-07-15/chkpt_model.27-acc0.95.hdf5"
    images_path = "/home/nyscf/Desktop/Classification_Model/data/test data"
    img_size = (400,400)
    num_images_to_load = 233
    test_batch_size = 1
    num_classes = 4
    WRITE_ON = True

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(images_path,
                                                    target_size=(img_size[0], img_size[1]),
                                                    batch_size=test_batch_size,
                                                    class_mode=None, 
                                                    shuffle = False)
    print("datagen created")


    model = load_model(model_path)

    if num_classes == 2:
        target_names = ['bad', 'good']
    if num_classes == 3:
        target_names = ['dead','diff', 'viable']
    if num_classes == 4:
        target_names = ['M1', 'M2', 'M3', 'M4']

    counter = 0
    for i in os.listdir(images_path):
        if i.endswith(".jpg") or i.endswith(".tiff"):
            print("reading", i)
            image = cv2.imread(images_path + "/" + i)
            out = image.copy()
            image = cv2.resize(image, img_size)
            image = image.astype("float") / 255.0
            images = [image]
            images = np.array(images)


            run_id, plate_id, well_id, date_taken = extract_attributes(i)
            
            prediction = model.predict(images, verbose=1, batch_size=1)
            print("Prediction for ", i, "= " , str(prediction))
            class_predicted = np.argmax(prediction)
            print("the class predicted is:", class_predicted)

            # If the probability for the most probable class is below a treshold, algorithm is uncertain and requires intervention
            if prediction[0][class_predicted] < 0.7 and class_predicted is not 0:
                cv2.imwrite(images_path + "/2019-07-15 model.27 predictions on EDRD0006/uncertain/" + i, out)
                continue

            if WRITE_ON:
                c = target_names[class_predicted] + " " + str(np.around(prediction[0][class_predicted], decimals=2))
                color = (255, 0, 0)
                cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imwrite(images_path + "/2019-07-15 model.27 predictions on EDRD0006/" + target_names[class_predicted] + "/" + i , out)
            certainty = str(np.around(prediction[0][class_predicted], decimals=2))
            

            row_in_csv = [counter, run_id, plate_id, well_id, date_taken, target_names[class_predicted], certainty, prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3]]

            with open(images_path + "/2019-07-15 model.27 predictions on EDRD0006/" + model_path.split("_")[-1] + "__predictions.csv", 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row_in_csv)

            counter += 1

    csvFile.close()