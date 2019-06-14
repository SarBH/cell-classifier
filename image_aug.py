#import all necessary packages
import os
import cv2
import numpy as np
import random
from PIL import Image

 
def rotate(image_path, degrees_to_rotate, saved_location, image_num):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
 
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location + 'rot180_' + str(image_num), 'TIFF')
    # rotated_image.show()
 

if __name__ == "__main__":
    directory = '/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/Original/'
    for image_num, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".tiff") or filename.endswith(".jpg"): 
            rotate(directory + filename, 180.0, '/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/rotate_180/', image_num)
            continue
        else:
            continue
