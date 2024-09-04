import os 
import matplotlib.pyplot as plt 
import pydicom as dicom
import numpy as np
from scipy.sparse import csc_matrix
import scipy.ndimage as scn
from collections import defaultdict
import os
import shutil
import operator
import warnings
import math
import medpy 
import medpy.metric
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from scipy.stats import scoreatpercentile

def preprocessing(path):
    '''
    参数：
        path: 文件夹路径,内含病人的ct图和标注图
    返回值：
        instanceUID2Imgname:字典,key是SOPInstanceUID, value是文件名
        Imgname2instanceUID:字典,key是文件名, value是key是SOPInstanceUID
    '''
    instanceUID2Imgname = {}
    Imgname2instanceUID = {}
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        f = dicom.read_file(img_path)

        if 'ROIContourSequence' not in dir(f):
            ct_data = f.pixel_array
            instanceUID = f.SOPInstanceUID
            if instanceUID not in instanceUID2Imgname:
                instanceUID2Imgname[instanceUID] = img_name
            else:
                print(f"BUG! {instanceUID} has two ct: {img_name} vs {instanceUID2Imgname[instanceUID]}")
            Imgname2instanceUID[img_name] = instanceUID
    
    return instanceUID2Imgname, Imgname2instanceUID


def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence 
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")

    return os.path.join(path, contour_file)

def get_roi_names(path):
    """
    This function will return the names of different contour data, 
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the 
    """
    contour_file = get_contour_file(path)
    contour_path = os.path.join(path, contour_file)
    contour_data = dicom.read_file(contour_path)
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def get_roi_index(roi_names, roi="CTV"):
    for index, roi_name in enumerate(roi_names):
        if roi_name == roi:
            return index 
    
    print(f"Error, There are no {roi} in {roi_names}")
    return None 


def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue
    slice_dict1 = {s.ImagePositionPatient[-1]: s.SOPInstanceUID for s in slices} #z轴坐标到uid的映射
    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices} #uid到z轴坐标的映射
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1)) #按z轴坐标排序的（uid，z轴坐标）元组列表
    return ordered_slices, slice_dict1


def calculate_percentile_hausdorff(binary_image1, binary_image2, percentile):
    """
    Calculate the percentile Hausdorff distance between two binary images.

    Parameters:
        binary_image1 (SimpleITK.Image): The first binary image.
        binary_image2 (SimpleITK.Image): The second binary image.
        percentile (float): The percentile of the Hausdorff distance to compute.

    Returns:
        float: The percentile Hausdorff distance.
    """
    # Convert SimpleITK images to NumPy arrays
    np_image1 = sitk.GetArrayFromImage(binary_image1)
    np_image2 = sitk.GetArrayFromImage(binary_image2)

    # Compute distance transform
    distance_map1 = distance_transform_edt(1 - np_image1)
    distance_map2 = distance_transform_edt(1 - np_image2)

    # Compute directed Hausdorff distances (from image1 to image2 and vice versa)
    dist1_to_2 = distance_map2[np_image1 == 1]
    dist2_to_1 = distance_map1[np_image2 == 1]

    # Combine distances
    all_distances = np.concatenate([dist1_to_2, dist2_to_1])

    # Compute the percentile
    hausdorff_distance_percentile = scoreatpercentile(all_distances, percentile)

    return hausdorff_distance_percentile


from scipy.interpolate import interp1d


def interpolation(volume, num_inserts=4):
    """
    对三维点阵volume在z轴上进行线性插值，每两个相邻的z层之间插入num_inserts张图。

    参数:
    - volume: 三维NumPy数组，形状为 (x, y, z)
    - num_inserts: 每两个z层之间要插入的图像数量

    返回:
    - 插值后的三维数组
    """
    # 获取输入volume的维度
    x_dim, y_dim, z_dim = volume.shape

    # 原始z坐标和新的z坐标
    original_z = np.arange(z_dim)
    new_z = np.linspace(0, z_dim - 1, z_dim + num_inserts * (z_dim - 1))
    print(len(new_z))

    # 创建一个新的数组来存储插值后的数据
    interpolated_volume = np.zeros((x_dim, y_dim, len(new_z)))

    # 对每个x, y位置上的z轴数据进行插值
    for x in range(x_dim):
        for y in range(y_dim):
            # 提取当前x, y位置上的z轴数据
            z_data = volume[x, y, :]

            # 创建插值模型
            interpolator = interp1d(original_z, z_data, kind='linear', fill_value="extrapolate")

            # 执行插值
            interpolated_volume[x, y, :] = interpolator(new_z)
    interpolated_volume = np.round(interpolated_volume)

    return interpolated_volume


def create_spherical_shell_mask(shape, center, radius, thickness=1):
    """
    创建一个球面mask。

    参数:
    - shape: 三维数组的形状，例如 (512, 512, 512)
    - center: 球心坐标，例如 (256, 256, 256)
    - radius: 球的半径
    - thickness: 球面的厚度（默认为1，即±0.5体素）

    返回:
    - 一个三维数组，其中球面区域内的元素为1，其他为0
    """
    # 初始化数组
    mask = np.zeros(shape, dtype=int)

    # 创建网格
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # 计算每个点到球心的距离
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)

    # 设置球面的位置为1，使用thickness定义球面厚度
    mask[np.abs(distance_from_center - radius) <= thickness / 2] = 1

    return mask


def create_spherical_mask(shape, center, radius):
    """
    创建一个球形mask。

    参数:
    - shape: 三维数组的形状，例如 (512, 512, 512)
    - center: 球心坐标，例如 (256, 256, 256)
    - radius: 球的半径

    返回:
    - 一个三维数组，其中球形区域内的元素为1，其他为0
    """
    # 初始化数组
    mask = np.zeros(shape, dtype=int)

    # 创建网格
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # 计算每个点到球心的距离
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)

    # 将球内的位置设置为1
    mask[distance_from_center <= radius] = 1

    return mask



if __name__ == '__main__':
    # 创建两个球面mask
    mask_shell_radius_50 = create_spherical_shell_mask((512, 512, 512), (256, 256, 256), 50)
    mask_shell_radius_100 = create_spherical_shell_mask((512, 512, 512), (256, 256, 256), 100)
    mask_radius_50 = create_spherical_mask((512, 512, 512), (256, 256, 256), 50)
    mask_radius_100 = create_spherical_mask((512, 512, 512), (256, 256, 256), 100)
    n_samples = 10000000

    # 球面测试完毕，试一下正方体
    # 创建一个全零的三维数组
    mask1 = np.zeros((512, 512, 512), dtype=int)
    mask2 = np.zeros((512, 512, 512), dtype=int)

    # 定义两个正方体的边长
    side_length_small = 50
    side_length_large = 100

    # 计算两个正方体的开始和结束索引
    start_small = 256 - side_length_small // 2
    end_small = 256 + side_length_small // 2
    start_large = 256 - side_length_large // 2
    end_large = 256 + side_length_large // 2

    # 填充较小的正方体的面
    mask1[start_small:end_small, start_small:end_small, start_small] = 1
    mask1[start_small:end_small, start_small:end_small, end_small - 1] = 1
    mask1[start_small:end_small, start_small, start_small:end_small] = 1
    mask1[start_small:end_small, end_small - 1, start_small:end_small] = 1
    mask1[start_small, start_small:end_small, start_small:end_small] = 1
    mask1[end_small - 1, start_small:end_small, start_small:end_small] = 1

    # 填充较大的正方体的面
    mask2[start_large:end_large, start_large:end_large, start_large] = 1
    mask2[start_large:end_large, start_large:end_large, end_large - 1] = 1
    mask2[start_large:end_large, start_large, start_large:end_large] = 1
    mask2[start_large:end_large, end_large - 1, start_large:end_large] = 1
    mask2[start_large, start_large:end_large, start_large:end_large] = 1
    mask2[end_large - 1, start_large:end_large, start_large:end_large] = 1

    mask1_itk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_itk = sitk.GetImageFromArray(mask2.astype(np.uint8))
    print(calculate_percentile_hausdorff(mask1_itk, mask2_itk, 50))
    print(calculate_percentile_hausdorff(mask1_itk, mask2_itk, 100))