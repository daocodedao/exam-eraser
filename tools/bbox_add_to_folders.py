from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os

#from y = 200 to y = 1470

def is_row_all_white(row, thresh, start, end):
    for i in range(start, end):
        if row[i] < thresh:
            return False
    return True

def is_col_all_white(pixels, col, thresh, top, bottom):
    if col == len(pixels[0]):
        return False
    for i in range(top, bottom):
        if pixels[i][col] < thresh:
            return False
    return True


def getboundaries(pixels, thresh, start, end, left, right, extension):
    res = []
    i = start
    while i < end:
        row = pixels[i]
        if not is_row_all_white(row, thresh, left, right):
            cur = []
            cur.append(i)
            start = i
            while i < end and (not is_row_all_white(row, thresh, left, right) or i - start < extension):
                i += 1
                if i >= end:
                    break
                row = pixels[i]
            cur.append(i)
            res.append(cur)
        i += 1
 
    return res

def get_all_vsplits(pixels, thresh, boundaries, left, right, extension):

    def get_vsplits(thresh, top, bottom, left, right, extension):
        
        res = []
        i = left
        while i < right:
            if not is_col_all_white(pixels, i, thresh, top, bottom):
                cur = []
                cur.append(i)
                start = i
                while not is_col_all_white(pixels, i, thresh, top, bottom) or i - start < extension:
                    if i >= right:
                        break
                    i += 1
                cur.append(i)
                res.append(cur)
            i += 1    
        
        return res

    splitmap = {}
    for up, down in boundaries:
        splitmap[(up, down)] =   get_vsplits(thresh, up, down, left, right, extension)
    
    return splitmap

def draw_bboxes(img, bboxes):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, fill=None, outline='RED')
    return img

def segment_bbox(pixels, bbox):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]

    boundaries = getboundaries(pixels, 210, top, bottom, left, right, 10)
    splitmap = get_all_vsplits(pixels, 210, boundaries, left, right, 20)

    bboxes = []
    for s in splitmap:
        split = splitmap[s]
        for sp in split:
            bboxes.append([sp[0], s[0], sp[1], s[1]])
            #format: left, top, right, bottom
    
    return bboxes

def find_black_line(pixels, bottom, right, thresh):

    def line_blackness(row):
        black = 0
        for i in range(0, right):
            if row[i] < thresh:
                black += 1
        return black / right
    
    for row in range(bottom - 1, 0, -1):
        lb = line_blackness(pixels[row])
        if lb > 0.40:
            return row
    return 1476 #general area of the line

def process_bbox_tiled_and_save(pixels, bbox, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    left, top, right, bottom = bbox
    crop = pixels[top:bottom, left:right]
    h, w = crop.shape

    tile_count = 0

    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = x + target_size
            y_end = y + target_size

            if x_end <= w and y_end <= h:
                # full size tile
                tile = crop[y:y_end, x:x_end]
            else:
                if h < target_size or w < target_size:
                    # padding when too small
                    tile = crop[y:min(y_end, h), x:min(x_end, w)]
                else:

                    if x_end > w and y_end > h:
                        sliver_ht = h % target_size
                        sliver_wd = w % target_size
                    elif x_end > w:
                        sliver_ht = target_size
                        sliver_wd = w % target_size
                    elif y_end > h:
                        sliver_ht = h % target_size
                        sliver_wd = target_size

                    y_start = max(0, y - (target_size - sliver_ht))
                    x_start = max(0, x - (target_size - sliver_wd))

                    tile = crop[y_start:min(y_end, h), x_start:min(x_end, w)]

                th, tw = tile.shape

                # pad tile to make it 20x20 if necessary
                pad_top = (target_size - th) // 2
                pad_bottom = target_size - th - pad_top
                pad_left = (target_size - tw) // 2
                pad_right = target_size - tw - pad_left

                tile = np.pad(
                    tile,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=255
                )

            chunk_left = left + x
            chunk_right = chunk_left + min(x_end, w)

            chunk_top = top + y
            chunk_bottom = chunk_top + min(y_end, h)

            base_name = str(chunk_left) + "_" + str(chunk_top) + "_" + str(chunk_right) + "_" + str(chunk_bottom)

            filename_with_coordinates = f"{base_name}.png"
            filepath = os.path.join(output_dir, filename_with_coordinates)

            tile_img = Image.fromarray(tile.astype(np.uint8))
            tile_img.save(filepath)

            tile_count += 1
        
            x += target_size
        y += target_size



def load(image_path):
    img = Image.open(image_path)
    pixels = np.array(img)

    COLS = len(pixels[0])
    ROWS = len(pixels)

    TOP_START = 200

    black_line = find_black_line(pixels, ROWS, COLS, 220) - 3

    boundaries = getboundaries(pixels, 210, TOP_START, black_line, 0, COLS, 0)
    splitmap = get_all_vsplits(pixels, 210, boundaries, 0, COLS, 30)

    bboxes = []
    for s in splitmap:
        split = splitmap[s]
        for sp in split:
            bboxes.append([sp[0], s[0], sp[1], s[1]])
            #format: left, top, right, bottom
    
    seg_bboxes = []
    for b in bboxes:
        seg_bboxes += segment_bbox(pixels, b)

    seg_bboxes_2 = []
    for b in seg_bboxes:
        seg_bboxes_2 += segment_bbox(pixels, b)
    
    output_folder = strname[:-4] + "_folder/"

    for bbox in seg_bboxes_2:
        process_bbox_tiled_and_save(pixels, bbox, output_folder, 20)

for i in range(1,8):
    strname = str(i) + ".jpg"
    print("processing",strname)
    load(strname)





    

    

