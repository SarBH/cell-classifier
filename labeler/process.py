import subprocess
import os
from datetime import datetime, timedelta
from collections import defaultdict

runs_dir = "/home/nyscf/Desktop/MMR_Runs"


def extract_attributes(image_name):

    # Extract attributes from name
    attr_list = image_name.split("_")
    run_id = attr_list[0]
    plate_id = attr_list[1] + "_" + attr_list[2]
    well_id = attr_list[6]
    date_taken = datetime.strptime(attr_list[3], '%m-%d-%Y-%I-%M-%S-%p')

    return (run_id, plate_id, well_id, date_taken)



for run_folder in os.listdir(runs_dir):

    for plate_date_folder in os.listdir(os.path.join(runs_dir, run_folder)):

        for image_name in os.listdir(os.path.join(runs_dir, run_folder, plate_date_folder)):           
            
            run_id, plate_id, well_id, date_taken = extract_attributes(image_name)
            
            
            # open current image
            print("opeining           |", run_id, "|", plate_id, "|", well_id, "|              date taken:", date_taken.strftime("%B %d %Y"))
            open_image_process = ["xdg-open", os.path.join(runs_dir, run_folder, plate_date_folder, image_name)]
            # out = subprocess.Popen(open_image_process, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            destination_dir = str("/home/nyscf/Desktop/MMR_Runs_by_Well/" + run_id + "__" + plate_id + "__" + well_id)
            os.makedirs(destination_dir)
            
            find_well_process = str('find ' + runs_dir + ' -name ' + '"' + run_id + '*"' + " -name " + '"*' + plate_id + '*"'+ " -name " + '"*' + well_id + '*"' + ' -exec cp {} ' + destination_dir + ' \;')
            print(find_well_process)
            # find_well_process = ["find ", os.path.join(runs_dir, run_folder), " | grep ", run_id, " | grep ", plate_id, " | grep ", well_id]
            proc = subprocess.Popen(find_well_process, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

            # P = ''.join(find_well_process)

            # proc = subprocess.Popen(P, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            # images_of_well = proc.stdout.read()
            # images_of_well = images_of_well.splitlines()

            # print(images_of_well)










"""
            # subprocess.call(["xdg-open", os.path.join(runs_dir, run_folder, plate_date_folder, image_name)])

            todo = input("what now?")
            if todo == "prev":
                date_seek = date_taken - timedelta(days=3)
                date_seek = date_seek.strftime("%m-%d-%Y")
                print(date_seek)
                well_seek = ["find ", os.path.join(runs_dir, run_folder), " | grep ", run_id, " | grep ", plate_id, " | grep ", well_id, " | grep ", date_seek]
                P = ''.join(well_seek)
                images_of_well = subprocess.call(P, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)
                if images_of_well == 1:
                    date_seek = date_taken - timedelta(days=4)
                    date_seek = date_seek.strftime("%m-%d-%Y")
                    well_seek = ["find ", os.path.join(runs_dir, run_folder), " | grep ", run_id, " | grep ", plate_id, " | grep ", well_id, " | grep ", date_seek]
                    P = ''.join(well_seek)
                    images_of_well = subprocess.call(P, shel
                # try:
                #     images_of_well = subprocess.(P, shell=
                # except subprocess.CalledProcessError as e:            print(images_of_well)

                #     raise RuntimeError("command '{}' return with error (code {}) {}:".format(e.cmd, e.returncode, e.output))
            # print("find", os.path.join(runs_dir, run_folder), "| grep", run_id, "| grep", plate_id, "| grep ", well_id)
            # find  /home/nyscf/Desktop/MMR_Runs/MMR0001 | grep MMR0001 | grep PS_103 | grep  A6
            # for img in $(find . | grep _PS_105 | grep _C9_); do xdg-open $img; done
            
             
            
"""