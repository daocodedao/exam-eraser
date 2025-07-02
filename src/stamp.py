from PIL import Image, ImageDraw
import json
import os
from pathlib import Path


def do_boxes_overlap(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)

def stamp_indiv(file,draw):
    box_int = [int(entry) for entry in file[:-4].split("_")]
    draw.rectangle(box_int, fill="white")

# 检查当前文件对应的矩形框是否与其他矩形框重叠
def overlap(file, data):
    # 将当前文件名称按 "_" 分割并转换为整数列表，作为当前矩形框的坐标
    box_int_self = [int(entry) for entry in file[:-4].split("_")]
    # 遍历数据中的每个键值对
    for key, value in data.items():
        # 排除当前文件，并且只处理值为 1 的情况
        if file != key and value == 1:
            # 将其他文件名称按 "_" 分割并转换为整数列表，作为其他矩形框的坐标
            box_int_other = [int(entry) for entry in key[:-4].split("_")]
            # 调用 do_boxes_overlap 函数检查两个矩形框是否重叠
            if do_boxes_overlap(box_int_self, box_int_other):
                return True
    return False

# 在原始图像上对符合条件的区域进行盖章处理
def stamp_all(original_image_path, folder):
    # 打开原始图像
    og_img = Image.open(original_image_path)
    # 创建一个 ImageDraw 对象，用于在图像上绘制
    draw = ImageDraw.Draw(og_img)

    # 定义保存结果图像的目录
    saveDir = "images/result/"
    # 创建保存目录，如果目录已存在则不会报错
    os.makedirs(saveDir, exist_ok=True)

    # 构建包含预测结果的 JSON 文件路径
    json_path = "temp/" + folder + "/predictions.json"

    # 打开 JSON 文件并加载其中的数据
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 遍历 JSON 数据中的每个键值对
    for key, value in data.items():
        # 如果值为 0 且当前矩形框不与其他矩形框重叠
        if value == 0 and not overlap(key, data):
            # 认为是手写内容，调用 stamp_indiv 函数进行盖章处理
            stamp_indiv(key,draw)

    file_name = Path(original_image_path).name
    save_path = saveDir + str(file_name)
    
    og_img.save(save_path)