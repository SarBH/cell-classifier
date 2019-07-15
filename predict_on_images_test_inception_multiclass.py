from keras.models import load_model
import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Activation

from keras import applications
from datetime import datetime
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



checkpoint_path = "/home/nyscf/Documents/sarita/models/inception_3_class/inception/model_2/chkpt_model.19-acc0.95.hdf5"
model_path = "/home/nyscf/Documents/sarita/models/inception_3_class/inception/model_inception_mb10_m5289_e50_c3.h5"
images_path = "/home/nyscf/Desktop/Training_Subsets_by_well/MMR0014_Colonies"

img_size = (400,400)
num_classes = 3
WRITE_ON = False


# FROM CHECKPOINT:
base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
                                                input_shape = (img_size[0], img_size[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)
	
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights(checkpoint_path)

model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ["acc"]) #used accuracy metric for 2 classes, supposedly catetgorical acc gets selected automatically when using cat cross ent

model.save(str(checkpoint_path[:-5] + ".h5"))




#FROM H5 MODEL:
# model = load_model(model_path)


counter = 0


for i in os.listdir(images_path):
    if i.endswith(".jpg") or i.endswith(".tiff"):
        image = cv2.imread(images_path + "/" + i)
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
        class_predicted = np.argmax(prediction)
        print("the class predicted is:", class_predicted)


        if class_predicted == 0:
            if WRITE_ON:
                c = "dead "+str(np.around(prediction[0][0], decimals=2))
                color = (255, 0, 0)
                cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(images_path + "/dead/" + i , out)
            class_predicted = 'dead'
            certainty = str(np.around(prediction[0][0], decimals=2))
        
        elif class_predicted == 1:
            if WRITE_ON:
                c = "diff "+str(np.around(prediction[0][1], decimals=2))
                color = (255, 0, 0)
                cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(images_path + "/diff/" + i , out)
            class_predicted = 'diff'
            certainty = str(np.around(prediction[0][1], decimals=2))

        else:
            if WRITE_ON:
                color = (0, 255, 0)
                c = "good "+str(np.around(prediction[0][2], decimals=2))
                cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(images_path + "/good/" + i, out)
            class_predicted = 'good'
            certainty = str(np.around(prediction[0][2], decimals=2))

        row_in_csv = [counter, run_id, plate_id, well_id, date_taken, class_predicted, certainty]

        with open(images_path + "/model_2_predictions.csv", 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row_in_csv)


        # cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imwrite("/home/nyscf/Desktop/labeled_colony_by_model/day9/all_labels/" + i + "__" + str(counter) + ".jpg", out)
        counter += 1
csvFile.close()