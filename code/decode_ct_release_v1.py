from medical import preprocessing, get_contour_file, slice_order, calculate_percentile_hausdorff, interpolation
from rt_utils import RTStructBuilder
import numpy as np
import medpy.metric
import SimpleITK as sitk
import os
import argparse
import time

def output(total_list, roi1, out_path):
    fr = open(out_path, 'a')
    for ans_dict in total_list:
        name = ans_dict["name"]
        IncI = ans_dict['IncI']
        CI = ans_dict['CI']
        RVD = ans_dict['RVD']
        mean_surface_distance = ans_dict['mean_surface_distance']
        centroid_distance = ans_dict['centroid_distance']
        dice_coefficient = ans_dict['dice_coefficient']
        hausdorff_distance_95 = ans_dict['hausdorff_distance_95']
        volume = ans_dict['volume']
        out_str = name + ',' + roi1 + ',' + str(IncI) + ',' + str(CI) + ',' + str(RVD) + ',' + str(mean_surface_distance) + ',' + str(centroid_distance) + ',' + str(dice_coefficient) + ',' + str(hausdorff_distance_95) + ',' + str(volume)
        print(out_str, file=fr)
    fr.close()
    copy_path = out_path[0:-4] + "_copy.csv"
    fr = open(copy_path, 'w')
    for ans_dict in total_list:
        name = ans_dict["name"]
        IncI = ans_dict['IncI']
        CI = ans_dict['CI']
        RVD = ans_dict['RVD']
        mean_surface_distance = ans_dict['mean_surface_distance']
        centroid_distance = ans_dict['centroid_distance']
        dice_coefficient = ans_dict['dice_coefficient']
        hausdorff_distance_95 = ans_dict['hausdorff_distance_95']
        volume = ans_dict['volume']
        out_str = name + ',' + roi1 + ',' + str(IncI) + ',' + str(CI) + ',' + str(RVD) + ',' + str(
            mean_surface_distance) + ',' + str(centroid_distance) + ',' + str(dice_coefficient) + ',' + str(
            hausdorff_distance_95) + ',' + str(volume)
        print(out_str, file=fr)
    fr.close()

def calc(mask_r1, mask_r2, tran_mat=np.array([1, 1, 1]), want_argv="all"):
    # 计算全部的指标
    # 计算Dice相似系数，下面有另一个包的，经过校验相等
    ans_dict = {}
    IncI = 0
    CI = 0
    RVD = 0
    mean_surface_distance = 0
    volume = 0
    centroid_distance = 0
    dice_coefficient = 0
    hausdorff_distance_95 = 0

    # 计算CI（concordance index）
    if want_argv == "all" or want_argv == "CI":
        coeff = medpy.metric.binary.dc(mask_r1, mask_r2)
        CI = coeff / (2 - coeff)

    # 计算IncI（inclusiveness index）
    if want_argv == "all" or want_argv == "IncI":
        intersection = np.sum(np.logical_and(mask_r1 > 0, mask_r2 > 0))
        IncI = intersection / (np.sum(mask_r2 == 1))

    # 相对体积差异relative volume difference, RVD
    if want_argv == "all" or want_argv == "RVD":
        RVD = (np.sum(mask_r1 == 1) - np.sum(mask_r2 == 1)) / (np.sum(mask_r2 == 1))

    # 计算质心偏差
    if want_argv == "all" or want_argv == "CD":
        def compute_centroid(mask):
            positions = np.where(mask == 1)
            return np.mean(positions, axis=1)
        centroid1 = compute_centroid(mask_r1)
        centroid2 = compute_centroid(mask_r2)
        centroid_distance = np.linalg.norm(np.multiply(centroid1 - centroid2, 1.0 / tran_mat))

    if want_argv == "all" or want_argv == "dc" or want_argv == "msd" or want_argv == "hd95":
        mask1_itk = sitk.GetImageFromArray(mask_r1.astype(np.uint8))
        mask2_itk = sitk.GetImageFromArray(mask_r2.astype(np.uint8))

    # 计算Dice相似系数
    if want_argv == "all" or want_argv == "dc":
        dice_filter = sitk.LabelOverlapMeasuresImageFilter()
        dice_filter.Execute(mask1_itk, mask2_itk)
        dice_coefficient = dice_filter.GetDiceCoefficient()

    # 计算平均表面距离
    if want_argv == "all" or want_argv == "msd":
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(mask1_itk, mask2_itk)
        mean_surface_distance = hausdorff_computer.GetAverageHausdorffDistance()

    # 计算绝对体积
    if want_argv == "all" or want_argv == "volume":
        volume = np.sum(mask_r1 == 1) / tran_mat[0] / tran_mat[1] / tran_mat[2]

    # 计算95距离
    if want_argv == "all" or want_argv == "hd95":
        hausdorff_distance_95 = calculate_percentile_hausdorff(mask1_itk, mask2_itk, 95)

    ans_dict['IncI'] = IncI
    ans_dict['CI'] = CI
    ans_dict['RVD'] = RVD
    ans_dict['mean_surface_distance'] = mean_surface_distance
    ans_dict['volume'] = volume
    ans_dict['centroid_distance'] = centroid_distance
    ans_dict['dice_coefficient'] = dice_coefficient
    ans_dict['hausdorff_distance_95'] = hausdorff_distance_95
    return ans_dict

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

        #print(f"mask_r1 shape is {mask_r1.shape}")  # 这里和之前的print是为了确认shape是否对齐了
        #print(f"mask_r2 shape is {mask_r2.shape}")

        new_mask_r1 = interpolation(mask_r1)
        new_mask_r2 = interpolation(mask_r2)
        return True, new_mask_r1, new_mask_r2
    except:
        return False, None, None


if __name__ == "__main__":
    # 加一个可以选算哪些指标的
    # 加一个表头
    # 去重做的鲁棒一些
    # 写两个一样的文件，有一个等效于备份
    # 优化写出的效率
    # 打印debug log
    description = """
    这个程序需要将student dictionary和teacher dictionary都放在该程序所在的路径下。
    程序用法：python3 decode_ct_release_v1.py --stu_path path1 --tea_path path2 --out_path path3 --stu_roi CTV --tea_roi CVT_SD --want_arv volume
    输出格式：第一行是表头，第二行开始是数据，目前已知以下两种情况会导致输出结果是0：①没有勾画对应的区域 ②指定了要计算的指标，则别的指标会填0
    """
    parser = argparse.ArgumentParser(description=description)

    # 添加参数
    parser.add_argument('--stu_path', default='./student', type=str, help='输入student所在的路径，默认是./student')
    parser.add_argument('--tea_path', default='./teacher', type=str, help='输入teacher所在的路径，默认是./teacher')
    parser.add_argument('--out_path', default='./result.csv', type=str, help='输入输出文件路径，请以.csv结尾，默认是./result.csv')
    parser.add_argument('--stu_roi', default='CTV', type=str, help='输入student_roi，默认是CTV')
    parser.add_argument('--tea_roi', default='CTV', type=str, help='输入teacher_roi，默认是CTV_SD')
    parser.add_argument('--want_arv', default='', type=str, help='输入需要算的指标，默认全算，支持的指标是IncI、CI、RVD、msd(mean_surface_distance)、volume、cd(centroid_distance)、dc(dice_coefficient)、hd95(hausdorff_distance_95)，请输入括号前的简写，而不是全称。如果想全算请不要填写这个参数或者填写all')
    # 解析命令行参数
    args = parser.parse_args()
    new_tran_mat = np.array([1.024, 1.024, 1])
    # 使用参数
    student_path = args.stu_path
    tea_path = args.tea_path
    out_path = args.out_path
    my_set = set([])
    if os.path.isfile(out_path):
        fr = open(out_path)
        for line in fr:
            key = line.strip().split(",")[0] + line.strip().split(",")[1]
            my_set.add(key)
        fr.close()
    roi1 = args.stu_roi
    roi2 = args.tea_roi
    want_arv = args.want_arv
    if not want_arv:
        want_arv = "all"
    total_list = []
    for _, tmp_student_path, _ in os.walk(student_path):
        for tmp in tmp_student_path:
            want_path = student_path + "/" + tmp
            print(want_path)
            tmp_key = want_path.split("/")[-1] + roi1
            if tmp_key in my_set:
                print(f"{tmp_key}已经算过了，我们跳过")
                continue
            print(f"开始计算{tmp_key}")
            start_time = time.perf_counter()
            flag, mask_r1, mask_r2 = main(want_path, tea_path, roi1=roi1, roi2=roi2, vision=False)
            end_time = time.perf_counter()
            ans_dict = {}
            ans_dict['IncI'] = 0
            ans_dict['CI'] = 0
            ans_dict['RVD'] = 0
            ans_dict['mean_surface_distance'] = 0
            ans_dict['volume'] = 0
            ans_dict['centroid_distance'] = 0
            ans_dict['dice_coefficient'] = 0
            ans_dict['hausdorff_distance_95'] = 0
            if flag:
                print("三维图构建成功！耗时{end_time - start_time}秒")
                start_time = time.perf_counter()
                ans_dict = calc(mask_r1, mask_r2, new_tran_mat, want_arv)
                end_time = time.perf_counter()
                print("计算指标完成！耗时{end_time - start_time}秒")
            print(f"{tmp_key}的结果是", ans_dict)
            ans_dict["name"] = tmp_key
            total_list.append(ans_dict)
    output(total_list, roi1, out_path)
































