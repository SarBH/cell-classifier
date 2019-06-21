import subprocess
import os
from datetime import datetime, timedelta
from collections import defaultdict

runs_dir = "/home/nyscf/Desktop/Training_Subets"


def extract_attributes(image_name):

    # Extract attributes from name
    attr_list = image_name.split("_")
    run_id = attr_list[0]
    plate_id = attr_list[1] + "_" + attr_list[2]
    well_id = attr_list[6]
    date_taken = datetime.strptime(attr_list[3], '%m-%d-%Y-%I-%M-%S-%p')

    return (run_id, plate_id, well_id, date_taken)

destination_dir_list = []

for run_folder in os.listdir(runs_dir):

    for clasification_type in os.listdir(os.path.join(runs_dir, run_folder)):

        for image_name in os.listdir(os.path.join(runs_dir, run_folder, clasification_type)):           
            
            run_id, plate_id, well_id, date_taken = extract_attributes(image_name)
            
            
            # identify current image
            print("exploring           |", clasification_type, "|", run_id, "|", plate_id, "|", well_id, "|              date taken:", date_taken.strftime("%B %d %Y"))
            # open_image_process = ["xdg-open", os.path.join(runs_dir, run_folder, plate_date_folder, image_name)]
            # out = subprocess.Popen(open_image_process, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            destination_dir = str("/home/nyscf/Desktop/Training_Subsets_by_well/" + clasification_type + "/" + run_id + "__" + plate_id + "__" + well_id)

            if destination_dir in destination_dir_list:
                print("A folder for this well has already been created. Skipping..")
                continue

            os.makedirs(destination_dir)
            
            find_well_process = str('find ' + os.path.join(runs_dir, run_folder, clasification_type) + ' -name ' + '"' + run_id + '*"' + " -name " + '"*' + plate_id + '*"'+ " -name " + '"*' + well_id + '*"' + ' -exec cp {} ' + destination_dir + ' \;')
            # print(find_well_process)
            # find_well_process = ["find ", os.path.join(runs_dir, run_folder), " | grep ", run_id, " | grep ", plate_id, " | grep ", well_id]
            proc = subprocess.Popen(find_well_process, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

            # P = ''.join(find_well_process)

            # proc = subprocess.Popen(P, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            # images_of_well = proc.stdout.read()
            # images_of_well = images_of_well.splitlines()

            # print(images_of_well)
            destination_dir_list.append(destination_dir)






