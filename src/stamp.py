from PIL import Image, ImageDraw
import json
import os

def do_boxes_overlap(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)

def stamp_indiv(file,draw):
    box_int = [int(entry) for entry in file[:-4].split("_")]
    draw.rectangle(box_int, fill="white")

def overlap(file, data):
    box_int_self = [int(entry) for entry in file[:-4].split("_")]
    for key, value in data.items():
        if file != key and value == 1:
            box_int_other = [int(entry) for entry in key[:-4].split("_")]
            if do_boxes_overlap(box_int_self, box_int_other):
                return True
    return False


def stamp_all(original_image, folder):
    original_image_path = "input-images/" + original_image
    og_img = Image.open(original_image_path)
    draw = ImageDraw.Draw(og_img)

    os.makedirs("stamped-images", exist_ok=True)

    json_path = "temp/" + folder + "/predictions.json"

    with open(json_path, 'r') as file:
        data = json.load(file)

    for key, value in data.items():
        if value == 0 and not overlap(key, data):
            #handwritten, stamp away
            stamp_indiv(key,draw)

    save_path = "stamped-images/" + str(original_image)
    
    og_img.save(save_path)