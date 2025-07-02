from PIL import Image
import numpy as np
import os
import uuid

# 从 y = 200 到 y = 1470

# 检查一行中的指定范围像素是否全为白色
def is_row_all_white(row, thresh, start, end):
    # 原始使用循环的检查方式，已注释掉
    # for i in range(start, end):
    #     if row[i] < thresh:
    #         return False
    # return True
    
    # 使用 NumPy 向量化操作检查指定范围的像素是否都大于等于阈值
    return np.all(row[start:end] >= thresh)

# 检查指定列中的指定范围像素是否全为白色
def is_col_all_white(pixels, col, thresh, top, bottom):
    # 原始使用循环的检查方式，已注释掉
    # if col == len(pixels[0]):
    #     return False
    # for i in range(top, bottom):
    #     if pixels[i][col] < thresh:
    #         return False
    # return True
    
    # 检查列索引是否超出图像宽度
    if col >= pixels.shape[1]:
        return False
    
    # 使用 NumPy 向量化操作检查指定列指定范围的像素是否都大于等于阈值
    return np.all(pixels[top:bottom, col] >= thresh)


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

# 查找图像中黑色线条所在的行号
def find_black_line(pixels, bottom, right, thresh):
    # 计算一行中黑色像素的比例
    def line_blackness(row):
        # 使用 NumPy 向量化操作计算黑色像素数量
        black_pixels = np.sum(row[:right] < thresh)
        # 返回黑色像素在该行中的比例
        return black_pixels / right
        # # 初始化黑色像素计数器
        # black = 0
        # # 遍历一行中的每个像素
        # for i in range(0, right):
        #     # 如果像素值小于阈值，则认为是黑色像素
        #     if row[i] < thresh:
        #         black += 1
        # # 返回黑色像素在该行中的比例
        # return black / right
    
    # 从图像底部向上遍历每一行
    for row in range(bottom - 1, 0, -1):
        # 计算当前行的黑色像素比例
        lb = line_blackness(pixels[row])
        # 如果黑色像素比例大于 0.40，则认为找到了黑色线条
        if lb > 0.40:
            return row
    # 若未找到符合条件的行，返回默认行号 1476
    return 1476 # 线条大致所在区域

def process_bbox_tiled_and_save(pixels, bbox, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    left, top, right, bottom = bbox
    crop = pixels[top:bottom, left:right]
    
    # Handle different number of channels
    if len(crop.shape) == 3:
        h, w, channels = crop.shape
    else:
        h, w = crop.shape
        channels = 1

    tile_count = 0

    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = x + target_size
            y_end = y + target_size

            if x_end <= w and y_end <= h:
                # Full size tile
                tile = crop[y:y_end, x:x_end]
            else:
                if h < target_size or w < target_size:
                    # Padding when too small
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

            if len(tile.shape) == 2:
                tile = tile[..., np.newaxis]

            th, tw, t_channels = tile.shape

            # Pad tile to make it 20x20 if necessary
            pad_top = (target_size - th) // 2
            pad_bottom = target_size - th - pad_top
            pad_left = (target_size - tw) // 2
            pad_right = target_size - tw - pad_left

            if t_channels == 1:
                tile = np.pad(
                    tile,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant',
                    constant_values=255
                )
            else:
                tile = np.pad(
                    tile,
                    pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant',
                    constant_values=255
                )

            chunk_left = left + x
            chunk_right = chunk_left + tile.shape[1]
            chunk_top = top + y
            chunk_bottom = chunk_top + tile.shape[0]

            base_name = str(chunk_left) + "_" + str(chunk_top) + "_" + str(chunk_right) + "_" + str(chunk_bottom)

            filename_with_coordinates = f"{base_name}.png"
            filepath = os.path.join(output_dir, filename_with_coordinates)

            tile_img = Image.fromarray(tile.squeeze().astype(np.uint8))
            tile_img.save(filepath)

            tile_count += 1
        
            x += target_size
        y += target_size

# 生成图像的边界框
def generate_bbox(image_path):
    # 打开指定路径的图像文件
    img = Image.open(image_path)
    # 将 PIL 图像对象转换为 NumPy 数组，方便后续进行数值计算
    pixels = np.array(img)

    # 获取图像的列数，即图像的宽度
    COLS = len(pixels[0])
    # 获取图像的行数，即图像的高度
    ROWS = len(pixels)

    # 定义起始行号，从第 200 行开始处理
    TOP_START = 200

    # 调用 find_black_line 函数查找图像中黑色线条所在的行号，并减去 3 作为边界
    black_line = find_black_line(pixels, ROWS, COLS, 220) - 3

    # 调用 getboundaries 函数获取图像中满足条件的水平边界
    boundaries = getboundaries(pixels, 210, TOP_START, black_line, 0, COLS, 0)
    # 调用 get_all_vsplits 函数根据水平边界获取垂直分割信息
    splitmap = get_all_vsplits(pixels, 210, boundaries, 0, COLS, 30)

    # 初始化一个空列表，用于存储生成的边界框
    bboxes = []
    # 遍历 splitmap 中的每个键值对
    for s in splitmap:
        # 获取当前键对应的垂直分割信息
        split = splitmap[s]
        # 遍历当前垂直分割信息中的每个分割结果
        for sp in split:
            # 将分割结果组合成边界框信息 [左边界, 上边界, 右边界, 下边界] 并添加到 bboxes 列表中
            bboxes.append([sp[0], s[0], sp[1], s[1]])
            #format: left, top, right, bottom
    
    # 初始化一个空列表，用于存储经过第一次细分后的边界框
    seg_bboxes = []
    # 遍历初始生成的边界框列表
    for b in bboxes:
        # 调用 segment_bbox 函数对每个边界框进行细分，将细分后的边界框添加到 seg_bboxes 列表中
        seg_bboxes += segment_bbox(pixels, b)

    # 初始化一个空列表，用于存储经过第二次细分后的边界框
    seg_bboxes_2 = []
    # 遍历第一次细分后的边界框列表
    for b in seg_bboxes:
        # 再次调用 segment_bbox 函数对每个边界框进行细分，将细分后的边界框添加到 seg_bboxes_2 列表中
        seg_bboxes_2 += segment_bbox(pixels, b)

    unique_name = uuid.uuid4()
    
    output_folder = "temp/" + str(unique_name) + "/"

    # 遍历经过两次细分后的边界框列表
    for bbox in seg_bboxes_2:
        # 调用 process_bbox_tiled_and_save 函数对每个边界框进行分块处理，并保存分块后的图像到指定输出文件夹
        # 目标分块大小为 20x20
        process_bbox_tiled_and_save(pixels, bbox, output_folder, 20)
    
    return unique_name




    

    

