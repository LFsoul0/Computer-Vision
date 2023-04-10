血管树部分代码启动文件为VascularTreeProject.py，
其中部分注释代码用于切换从头开始/读取中间结果文件，
默认读取当前文件夹中的data/case_1/DICOM，默认输出到output/case_1
使用命令行参数--input和--output可分别指定输入输出路径

关于输出
两个case的输出均在output文件夹中，
其中nii文件为每个case的最终结果，
其余为中间文件，可用于三维重建等