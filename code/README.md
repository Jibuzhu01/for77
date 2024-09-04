一、软件安装
    1、安装anaconda最新版本
    2、命令行运行：pip install medpy, pydicom, matplotlib 

二、函数说明
    1、核心的运行函数是rts.py下的main函数
        def main(path1, path2, roi1="CTV_SD", roi2="CTV_PRE", vision=False):

        该函数传入两个文件夹，以及需要比对的区域roi1,roi2

        这两个文件夹就是要比对的两个人，比如path1 = "case4/lidonghua-04",path2 = "case4/PUMCH04-Standard-79"

        roi1对应的是path1的比对区域,roi2对应的是path2的比对区域，计算结果就是这两个文件夹对应的dice系数

        注意一个文件夹只能有一个rs文件和全套的ct图，不符合这个格式的都有可能报错

        如果要查看当前比对的所有mask的图, 参数vision=True

        解析结果:按照z坐标命名的文件夹下有每一对mask的图

    2、大批量计算数据的时候调用main1函数,一个简单的循环,由于大批量计算格式不稳定，因此该函数需要根据实际情况改写
