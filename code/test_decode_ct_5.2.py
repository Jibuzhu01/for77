from medical import preprocessing, get_contour_file, slice_order, calculate_percentile_hausdorff, interpolation
from rt_utils import RTStructBuilder
import numpy as np
import medpy.metric
import SimpleITK as sitk
import os

def output(name, roi1, ans_dict, out_path, flag):
    fr = open(out_path, 'a')
    if flag:
        IncI = ans_dict['IncI']
        CI = ans_dict['CI']
        RVD = ans_dict['RVD']
        mean_surface_distance = ans_dict['mean_surface_distance']
        centroid_distance = ans_dict['centroid_distance']
        dice_coefficient = ans_dict['dice_coefficient']
        hausdorff_distance_95 = ans_dict['hausdorff_distance_95']
        volume = ans_dict['volume']
        out_str = name + ',' + roi1 + ',' + str(IncI) + ',' + str(CI) + ',' + str(RVD) + ',' + str(mean_surface_distance) + ',' + str(centroid_distance) + ',' + str(dice_coefficient) + ',' + str(hausdorff_distance_95) + ',' + str(volume)
    else:
        out_str = name + ',' + roi1 + ',' + "0,0,0,0,0,0,0,0"
    print(out_str, file=fr)
    fr.close()

def calc(mask_r1, mask_r2, tran_mat=np.array([1, 1, 1])):
    # 计算全部的指标
    # 计算Dice相似系数，下面有另一个包的，经过校验相等
    coeff = medpy.metric.binary.dc(mask_r1, mask_r2)

    # 计算CI（concordance index）
    CI = coeff / (2 - coeff)

    # 计算IncI（inclusiveness index）
    intersection = np.sum(np.logical_and(mask_r1 > 0, mask_r2 > 0))
    IncI = intersection / (np.sum(mask_r2 == 1))
    print(f"IncI is: {IncI}")

    # 相对体积差异relative volume difference, RVD
    RVD = (np.sum(mask_r1 == 1) - np.sum(mask_r2 == 1)) / (np.sum(mask_r2 == 1))
    #print(f"Dice coeff is: {coeff}")
    print(f"CI is: {CI}")
    print(f"RVD is: {RVD}")

    # 计算质心偏差
    def compute_centroid(mask):
        positions = np.where(mask == 1)
        return np.mean(positions, axis=1)
    centroid1 = compute_centroid(mask_r1)
    centroid2 = compute_centroid(mask_r2)
    centroid_distance = np.linalg.norm(np.multiply(centroid1 - centroid2, 1.0 / tran_mat))
    print(f'Centroid Distance: {centroid_distance}')

    # 计算Dice相似系数
    mask1_itk = sitk.GetImageFromArray(mask_r1.astype(np.uint8))
    mask2_itk = sitk.GetImageFromArray(mask_r2.astype(np.uint8))
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter.Execute(mask1_itk, mask2_itk)
    dice_coefficient = dice_filter.GetDiceCoefficient()
    print(f'Dice Similarity Coefficient: {dice_coefficient}')

    # 计算平均表面距离
    hausdorff_computer = sitk.HausdorffDistanceImageFilter()
    hausdorff_computer.Execute(mask1_itk, mask2_itk)
    mean_surface_distance = hausdorff_computer.GetAverageHausdorffDistance()
    print(f'Mean Surface Distance: {mean_surface_distance}')

    # 计算绝对体积
    volume = np.sum(mask_r1 == 1) / tran_mat[0] / tran_mat[1] / tran_mat[2]
    n_samples = 100000000

    # 获得mask的尺寸
    x_size, y_size, z_size = mask_r2.shape

    # 随机生成采样点
    x = np.random.randint(0, x_size, n_samples)
    y = np.random.randint(0, y_size, n_samples)
    z = np.random.randint(0, z_size, n_samples)
    inside_points = np.sum(mask_r2[x, y, z])
    total_volume = inside_points / n_samples * x_size * y_size * z_size / tran_mat[0] / tran_mat[1] / tran_mat[2]
    print(f'volume: {volume}')
    print(f'total_volume: {total_volume}')

    # 计算95距离
    hausdorff_distance_95 = calculate_percentile_hausdorff(mask1_itk, mask2_itk, 95)
    print(f'95% Hausdorff Distance: {hausdorff_distance_95}')
    ans_dict = {}
    ans_dict['IncI'] = IncI
    ans_dict['CI'] = CI
    ans_dict['RVD'] = RVD
    ans_dict['mean_surface_distance'] = mean_surface_distance
    ans_dict['volume'] = volume
    ans_dict['centroid_distance'] = centroid_distance
    ans_dict['dice_coefficient'] = dice_coefficient
    ans_dict['hausdorff_distance_95'] = hausdorff_distance_95
    return ans_dict
    #output(name, roi1, IncI, CI, RVD, mean_surface_distance, centroid_distance, dice_coefficient, hausdorff_distance_95, out_path)

def main(path1, path2, roi1="CTV_PRE", roi2="CTV_SD", vision=False):
    #instanceUID2Imgname1, Imgname2instanceUID1 = preprocessing(path1)
    #instanceUID2Imgname2, Imgname2instanceUID2 = preprocessing(path2)

    flag = True
    ans_dict = {}
    try:
        contour_file1 = get_contour_file(path1)
        contour_file2 = get_contour_file(path2)

        print(f"contour_file1 is {contour_file1}, contour_file2 is {contour_file2}")

        rtstruct1 = RTStructBuilder.create_from(dicom_series_path=path1, rt_struct_path=contour_file1)
        rtstruct2 = RTStructBuilder.create_from(dicom_series_path=path2, rt_struct_path=contour_file2)

        roi_names1 = rtstruct1.get_roi_names()
        roi_names2 = rtstruct2.get_roi_names()

        print(f"{path1} roi is: {roi_names1}")
        print(f"{path2} roi is: {roi_names2}")

        if roi1 not in roi_names1:
            print(f"Error, {path1} does not have {roi1}, can not compute dice")

        if roi2 not in roi_names2:
            print(f"Error, {path2} does not have {roi2}, can not compute dice")

        mask_3d1, instanceUIDs1 = rtstruct1.get_roi_mask_by_name(roi1)
        mask_3d2, instanceUIDs2 = rtstruct2.get_roi_mask_by_name(roi2)

        print(f"mask_3d1 shape is {mask_3d1.shape}")
        print(f"mask_3d2 shape is {mask_3d2.shape}")

        uid2mask1, uid2mask2 = {}, {}
        for i, uid in enumerate(instanceUIDs1):
            uid2mask1[uid] = mask_3d1[:, :, i]

        for i, uid in enumerate(instanceUIDs2):
            uid2mask2[uid] = mask_3d2[:, :, i]

        ordered_slices1, slice_dict1 = slice_order(path1)
        ordered_slices2, slice_dict2 = slice_order(path2)

        z_positions = list(slice_dict1.keys() & slice_dict2.keys())  # z轴坐标交集
        z_positions = sorted(z_positions)

        height, width, channel = mask_3d1.shape  # channel其实没用到，标准写法是_
        mask_r1 = np.zeros((height, width, len(z_positions)))  # 重新构造mask三维点阵，只保留z轴有交集的部分
        mask_r2 = np.zeros((height, width, len(z_positions)))

        for i, z in enumerate(z_positions):
            mask_r1[:, :, i] = uid2mask1[slice_dict1[z]]  # 双重字典读取，首先通过z轴坐标找到uid，然后通过uid对应到原始mask层
            mask_r2[:, :, i] = uid2mask2[slice_dict2[z]]

        print(f"mask_r1 shape is {mask_r1.shape}")  # 这里和之前的print是为了确认shape是否对齐了
        print(f"mask_r2 shape is {mask_r2.shape}")

        new_mask_r1 = interpolation(mask_r1)
        new_mask_r2 = interpolation(mask_r2)
        new_tran_mat = np.array([1.024, 1.024, 1])
        ans_dict = calc(new_mask_r1, new_mask_r2, new_tran_mat)
    except:
        flag = False
    name = path1.split("/")[-1]
    output(name, roi1, ans_dict, out_path, flag)


if __name__ == "__main__":
    # 加一个可以选算哪些指标的
    # 加一个表头
    # 去重做的鲁棒一些
    # 写两个一样的文件，有一个等效于备份
    # 优化写出的效率
    # 打印debug log
    student_path = "/Users/guqiqi/Downloads/Aug/case5"
    tea_path = "/Users/guqiqi/Downloads/st/PUMCH05-Standard"
    out_path = "/Users/guqiqi/Downloads/coderesult/Augcase5.2.csv"
    my_set = set([])
    try:
        fr = open(out_path)
        for line in fr:
            key = line.strip().split(",")[0] + line.strip().split(",")[1]
            my_set.add(key)
        fr.close()
    except:
        pass
    roi1 = "CTV"
    roi2 = "CTV_SD"
    for _, tmp_student_path, _ in os.walk(student_path):
        for tmp in tmp_student_path:
            want_path = student_path + "/" + tmp
            print(want_path)
            tmp_key = want_path.split("/")[-1] + roi1
            if tmp_key in my_set:
                continue
            main(want_path, tea_path, roi1=roi1, roi2=roi2, vision=False)































