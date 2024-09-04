from medical import preprocessing, get_contour_file, slice_order
from rt_utils import RTStructBuilder
import numpy as np  
import medpy.metric
import matplotlib.pyplot as plt
import os 


def main(path1, path2, roi1="CTV_PRE", roi2="CTV_SD", vision=False):
	instanceUID2Imgname1, Imgname2instanceUID1 = preprocessing(path1)
	instanceUID2Imgname2, Imgname2instanceUID2 = preprocessing(path2)

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

	# 到这里，mask_3d1里面存了立体灰度图，图像大小是512*512*106，前面两个512是从CT图中读取的参数，最后106是z轴切片的数量，是实际有勾画区域切片范围的超集
	# 这里instanceUIDs1、instanceUIDs2存储的是ct图像id，可以认为是一个平面的唯一id，所以是不重叠的并且可以和z轴坐标一一对应
	# 接下来我们要进行变换，这里的目的是通过uid对齐不同的ct图片张数，之前导出的有两种不同的切片总数，如果导出的完全匹配可以忽略下面的这个复杂变换
	# 首先做一个字典，以uid（ct图像的唯一id）为key存储mask图像

	uid2mask1, uid2mask2 = {}, {}
	for i, uid in enumerate(instanceUIDs1):
		uid2mask1[uid] = mask_3d1[:, :, i]

	for i, uid in enumerate(instanceUIDs2):
		uid2mask2[uid] = mask_3d2[:, :, i]


	ordered_slices1, slice_dict1 = slice_order(path1)
	ordered_slices2, slice_dict2 = slice_order(path2)

	z_positions = list(slice_dict1.keys() & slice_dict2.keys()) #z轴坐标交集
	z_positions = sorted(z_positions)
	

	height, width, channel = mask_3d1.shape #channel其实没用到，标准写法是_
	mask_r1 = np.zeros((height, width, len(z_positions))) #重新构造mask三维点阵，只保留z轴有交集的部分
	mask_r2 = np.zeros((height, width, len(z_positions)))

	for i, z in enumerate(z_positions):
		mask_r1[:, :, i] = uid2mask1[slice_dict1[z]] #双重字典读取，首先通过z轴坐标找到uid，然后通过uid对应到原始mask层
		mask_r2[:, :, i] = uid2mask2[slice_dict2[z]]

	print(f"mask_r1 shape is {mask_r1.shape}") #这里和之前的print是为了确认shape是否对齐了
	print(f"mask_r2 shape is {mask_r2.shape}")

	coeff = medpy.metric.binary.dc(mask_r1, mask_r2)
	CI = coeff / (2 - coeff)
	RVD = (np.sum(mask_r1 == 1) - np.sum(mask_r2 == 1)) / (np.sum(mask_r2 ==1)) #相对体积差异relative volume difference, RVD
	print(f"Dice coeff is: {coeff}")
	print(f"CI is: {CI}")
	print(f"RVD is: {RVD}")

	if vision:
		os.makedirs("rts-res", exist_ok=True)

		with open("rts-res/coeff_total.txt", "w") as f1:
			f1.write(str(coeff) + "\n")

		for z in z_positions:
			dir1 = os.path.join("rts-res", str(z))
			os.makedirs(dir1, exist_ok=True)

			m1 = uid2mask1[slice_dict1[z]]
			m2 = uid2mask2[slice_dict2[z]]

			name1 = instanceUID2Imgname1[slice_dict1[z]].replace(".dcm", ".jpg")
			name2 = instanceUID2Imgname2[slice_dict2[z]].replace(".dcm", ".jpg")

			img_p1 = os.path.join(dir1, name1)
			img_p2 = os.path.join(dir1, name2)

			plt.imsave(img_p1, m1)
			plt.imsave(img_p2, m2)

			c1 = medpy.metric.binary.dc(m1, m2)

			with open(f"{dir1}/coeff.txt", "w") as f2:
				f2.write(str(c1) + "\n")

	return coeff


def main1(dirs):
	standard_dirs, sample_dirs = [], []
	for dir1 in os.listdir(dirs):
		if "standard" in dir1 or "Standard" in dir1:
			standard_dirs.append(dir1)
		else:
			sample_dirs.append(dir1)

	print(f"standard_dirs is {standard_dirs}, sample_dirs is {sample_dirs}")

	roi1s = ["CTV_SD", "CTV_S_SD", "CTVC_SD", "CTVU_SD", "CTVNa_SD", "CTVNp_SD"]

	with open(f"{dirs}-res.txt", "a") as f: 
		for standard_dir in standard_dirs:
			standard_path = os.path.join(dirs, standard_dir)
			for sample_dir in sample_dirs:
				sample_path = os.path.join(dirs, sample_dir)

				for roi1 in roi1s:
					roi2 = roi1[:-2] + "PRE"
					print(f"{standard_dir} {roi1} vs {sample_dir} {roi2}")
					try:
						coeff = main(standard_path, sample_path, roi1, roi2)
						f.write(standard_dir + "\t" + roi1 + "\t" + sample_dir + "\t" + roi2 + "\t" + str(coeff) + "\n")
					except:
						print(f"Crash! {standard_dir} {roi1} vs {sample_dir} {roi2}")
						continue 



if __name__ == "__main__":
	student_path = "/Users/bytedance/Downloads/case5/PUMCH05-LiuDongbin"
	tea_path = "/Users/bytedance/Downloads/case5/PUMCH05-Standard"

	main(student_path, tea_path, vision=True)
	#dirs = "case4"
	#main1(dirs)


