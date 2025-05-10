from bbox_bounds import generate_bbox
from inference import classify_all
from architecture import *
from tqdm import tqdm, trange
from stamp import stamp_all
import shutil

import os
import img2pdf
import sys

def main():
    
    def load(strname):
        image_name = "input-images/" + strname
        image_folder = str(generate_bbox(image_name))

        image_name_folder_map[strname] = image_folder
    
    input_folder = sys.argv[1]

    if not os.path.isdir(input_folder):
        print("Error: Can't find the input folder " + input_folder)
        return

    image_name_folder_map = {}

    for root, dirs, files in os.walk("input-images"):
        for file in tqdm(files, desc="Getting bouding boxes"):
            load(file)

    classify_all(image_name_folder_map)

    for k, v in tqdm(image_name_folder_map.items(), desc="Stamping"):
        stamp_all(k, v)

    imp = []
    for root, dirs, files in os.walk("stamped-images"):
        imp = files
    
    for i in range(len(imp)):
        imp[i] = "stamped-images/" + imp[i]


    with open("processed.pdf","wb") as f:
        f.write(img2pdf.convert(imp))
    
    shutil.rmtree('temp')
    shutil.rmtree('stamped-images')

main()