from bbox_bounds import generate_bbox
from inference import classify_all
from architecture import *
from tqdm import tqdm, trange
from stamp import stamp_all
import shutil

import os
import img2pdf
import sys

def main(input_folder="images/"):
    # 若命令行提供参数，则覆盖默认的输入文件夹路径
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]

    def load(imagePath):
        # 调用 generate_bbox 函数生成图像的边界框文件夹路径，并转换为字符串
        image_folder = str(generate_bbox(imagePath))
        # 将图像路径与对应的边界框文件夹路径关联起来，存储到字典中
        image_name_folder_map[imagePath] = image_folder

    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print("Error: Can't find the input folder " + input_folder)
        return

    # 初始化一个空字典，用于存储图像路径与对应的边界框文件夹路径的映射关系
    image_name_folder_map = {}
    # 定义支持的图像文件扩展名元组
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    # 递归遍历输入文件夹及其子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 使用 tqdm 显示进度条，遍历当前文件夹中的所有文件
        for file in tqdm(files, desc="Getting bouding boxes"):
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            # 检查文件是否为支持的图像文件
            if file.lower().endswith(IMAGE_EXTENSIONS):
                # 调用 load 函数处理图像文件
                load(file_path)

    # 调用 classify_all 函数对所有图像进行分类
    classify_all(image_name_folder_map)

    # 使用 tqdm 显示进度条，遍历图像路径与边界框文件夹路径的映射关系
    for k, v in tqdm(image_name_folder_map.items(), desc="Stamping"):
        # 调用 stamp_all 函数对图像进行盖章处理
        stamp_all(k, v)

    # 定义保存结果图像的文件夹路径
    saveDir = "images/result/"
    # 初始化一个空列表，用于存储结果图像文件的名称
    imp = []
    # 递归遍历保存结果图像的文件夹及其子文件夹
    for root, dirs, files in os.walk(saveDir):
        # 将结果图像文件的名称存储到列表中
        imp = files

    # 遍历结果图像文件名称列表
    for i in range(len(imp)):
        # ... 此处代码未完整，推测是要处理结果图像文件的路径 ...
        imp[i] = saveDir + imp[i]

    # 以二进制写入模式打开 processed.pdf 文件
    with open("processed.pdf","wb") as f:
        # 将结果图像文件转换为 PDF 格式并写入文件
        f.write(img2pdf.convert(imp))

    # 删除临时文件夹
    shutil.rmtree('temp')
    # 删除保存结果图像的文件夹
    shutil.rmtree(saveDir)

main()