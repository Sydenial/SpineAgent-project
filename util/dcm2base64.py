import os
from typing import Tuple, List

import pydicom
import numpy as np
from PIL import Image
import io
import base64
from collections import defaultdict


def is_sagittal(ds, tolerance=0.5):
    """判断是否为矢状位"""
    if "ImageOrientationPatient" not in ds:
        return False
    v1 = np.array(ds.ImageOrientationPatient[:3])
    v2 = np.array(ds.ImageOrientationPatient[3:])
    normal = np.cross(v1, v2)
    # 法向量主要在 X 轴上 (|x| 接近 1)
    return abs(normal[0]) > tolerance


def get_8bit_image(ds):
    """处理 DICOM 像素，应用窗宽窗位并转为 8 位"""
    pixel_array = ds.pixel_array.astype(float)

    # 应用 Rescale (Slope/Intercept)
    rescale_slope = getattr(ds, 'RescaleSlope', 1)
    rescale_intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * rescale_slope + rescale_intercept

    # 获取窗宽窗位
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
    else:
        wc, ww = np.percentile(pixel_array, [50, 95])  # 自动估算

    val_min = wc - ww / 2
    val_max = wc + ww / 2

    pixel_array = np.clip(pixel_array, val_min, val_max)
    pixel_array = ((pixel_array - val_min) / (val_max - val_min) * 255.0).astype(np.uint8)
    return Image.fromarray(pixel_array)


def process_uploaded_dicoms_to_base64(uploaded_files) -> tuple[list[list[str]], list[list[str]]]:
    """
    处理前端上传的 DICOM 文件流，提取中间 5 层的矢状位图像，
    直接返回 Base64 字符串列表，供 LangChain 调用。
    """
    pos_groups = defaultdict(list)

    # 1. 直接读取内存中的文件流
    for uploaded_file in uploaded_files:
        try:
            # 读取字节流
            file_bytes = uploaded_file.read()
            ds = pydicom.dcmread(io.BytesIO(file_bytes))

            if is_sagittal(ds):
                x_pos = round(ds.ImagePositionPatient[0], 2)
                # 直接将 dataset 存入字典，避免二次读取
                pos_groups[x_pos].append(ds)
        except Exception as e:
            print(f"读取上传文件失败: {e}")

    if not pos_groups:
        raise ValueError("上传的影像中未找到矢状位图像。")

    # 2. 去除定位像
    counts = [len(v) for v in pos_groups.values()]
    if any(c > 1 for c in counts):
        valid_groups = {k: v for k, v in pos_groups.items() if len(v) > 1}
    else:
        valid_groups = pos_groups

    # 3. 按位置排序并只保留中间 5 个位置
    sorted_x = sorted(valid_groups.keys())
    num_pos = len(sorted_x)

    if num_pos <= 5:
        target_x = sorted_x
    else:
        mid_idx = num_pos // 2
        target_x = sorted_x[mid_idx - 2: mid_idx + 3]

    # 4. 按位置分组，转换为二维列表按位置分组，转换为二维列表
    grouped_base64_images = []
    grouped_sequence_names = []

    for x in target_x:
        current_pos_images = []  # 存放当前这 1 个切片位置的所有序列图片
        current_pos_seqs = []  # 存放当前这 1 个切片位置的所有序列名称

        for ds in valid_groups[x]:
            img = get_8bit_image(ds)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 读取序列名称
            seq_name = str(getattr(ds, 'ProtocolName', getattr(ds, 'SeriesDescription', '未知序列')))

            current_pos_images.append(b64_str)
            current_pos_seqs.append(seq_name)

        # 将当前位置的图片组和名称组作为一个整体（列表），追加到大列表中
        grouped_base64_images.append(current_pos_images)
        grouped_sequence_names.append(current_pos_seqs)

    return grouped_base64_images, grouped_sequence_names
