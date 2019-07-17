from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
import numpy as np

def crop_edges(img):
    """
    Crops black edges from a full Celigo image
    Must be read in grayscale (single-channel)
    """
    imarray = np.array(img)
    slideIndex = [0, len(imarray) - 1, 0, len(imarray[0]) - 1]
    left_indent, top_indent, right_indent, bottom_indent = [0, 0, 0, 0]
    pixel_threshold = 70
    while np.max(imarray[slideIndex[0]]) <= pixel_threshold:
      top_indent += 1
      slideIndex[0] += 1
    while np.max(imarray[slideIndex[1]]) <= pixel_threshold:
      bottom_indent += 1
      slideIndex[1] -= 1
    while np.max(imarray.T[slideIndex[2]]) <= pixel_threshold:
      left_indent += 1
      slideIndex[2] += 1
    while np.max(imarray.T[slideIndex[3]]) <= pixel_threshold:
      right_indent += 1
      slideIndex[3] -= 1

    slidedImarray = imarray[
      slideIndex[0]: slideIndex[1],
      slideIndex[2]: slideIndex[3]]

    indents = [left_indent, top_indent, right_indent, bottom_indent]

    # Returning slide index allows us to keep track of how far the image was cropped
    return [slidedImarray, indents]

pld_model_path = '/home/nyscf/Documents/sarita/cell-classifier/preprocessing/brodie/multi_class_v1-1_epoch12.h5'

pld_model = models.load_model(pld_model_path, backbone_name='resnet50')

imagelist = [i.strip() for i in open("/home/nyscf/Documents/sarita/cell-classifier/preprocessing/brodie/MMR0028_copy_102_104_106_7-15-2019_file_names_v1.txt")]


c = 0
t = 0
for i in imagelist:
    print ("Reading " + i.split("/")[-1])
    file_name = i.split("/")[-1]
    prefix = i.split("/")[:-1]
    img_path = "/".join(prefix) + "/" + file_name.split("__")[-1]
    img = cv2.imread(img_path, 0)
    img, base_coords = crop_edges(img)
    draw = img.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = preprocess_image(img)
    img, scale = resize_image(img)

    boxes, scores, labels = pld_model.predict_on_batch((np.expand_dims(img, axis=0)))

    boxes /= scale
    boxes = boxes.astype(int)

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break
        print("Found something, saving..")
        x1, y1, x2, y2 = box
        if label == 0:
            d2 = draw.copy()
            d2 = d2[y1:y2, x1:x2]
            cv2.imwrite("/home/nyscf/Desktop/Training_Subets/MMR0028_copy_102_104_106_7-15-2019/" + file_name.split(".")[0] + str(x1) + "--" +
                        str(y1) + "--" + str(x2) + "--" + str(y2) + ".jpg", d2)
        elif label == 2:
            d2 = draw.copy()
            d2 = d2[y1:y2, x1:x2]
            cv2.imwrite("/home/nyscf/Desktop/Training_Subets/MMR0028_copy_102_104_106_7-15-2019/" + file_name.split(".")[0] + str(x1) + "--" +
                        str(y1) + "--" + str(x2) + "--" + str(y2) + ".jpg", d2)
        elif label == 1:
            d2 = draw.copy()
            d2 = d2[y1:y2, x1:x2]
            cv2.imwrite("/home/nyscf/Desktop/Training_Subets/MMR0028_copy_102_104_106_7-15-2019/" + file_name.split(".")[0] + str(x1) + "--" +
                        str(y1) + "--" + str(x2) + "--" + str(y2) + ".jpg", d2)
