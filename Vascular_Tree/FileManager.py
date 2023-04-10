import pydicom
import nibabel as nib
import numpy as np
import cv2
import glob
import os

def ReadDicomDir(path : str, layer_shape : tuple) -> np.ndarray :
    if len(layer_shape) != 2 :
        print("Read file error : layer_shape must be tuple like (x, y)")
        return np.empty((0, 0, 0), np.int16)

    dicom_files = glob.glob(path + '/*.dcm')

    # DICOM和NIFTI的层数关系相反，即 DICOM 第一层对应 NIFTI 最后一层
    dicom_files.sort(key=lambda s : int(
        s.split('\\')[-1].split('/')[-1].split('-')[-1].split('.dcm')[0]), reverse = True)  # macOS / Linux 可能需要以 '/' 分隔路径

    result = np.empty(layer_shape + (len(dicom_files),), np.int16);

    for j in range(len(dicom_files)):
        dicom = pydicom.read_file(dicom_files[j])

        result[:, :, j] = dicom.pixel_array.astype(np.int16).T
        result[:, :, j] *= np.int16(dicom.RescaleSlope)
        result[:, :, j] += np.int16(dicom.RescaleIntercept)

    return result


def ReadPNGDir(path : str, layer_shape : tuple) -> np.ndarray :
    if len(layer_shape) != 2 :
        print("Read file error : layer_shape must be tuple like (x, y)")
        return np.empty((0, 0, 0), np.uint8)

    png_files = glob.glob(path + '/*.png')

    # DICOM和NIFTI的层数关系相反，即 DICOM 第一层对应 NIFTI 最后一层
    png_files.sort(key=lambda s : int(
        s.split('\\')[-1].split('/')[-1].split('.png')[0]), reverse = False)  # macOS / Linux 可能需要以 '/' 分隔路径

    result = np.empty(layer_shape + (0,), np.uint8);

    for j in range(len(png_files)):
        img = cv2.imread(png_files[j], cv2.IMREAD_GRAYSCALE).astype(np.uint8).T

        img = img[:, :, np.newaxis];
        result = np.append(result, img, axis = 2)

    return result


def Write3DArrayAsImages(path : str, data : np.ndarray) :
    if len(data.shape) != 3 :
        print("Write file error: data must be a 3D-array")
        return

    if not os.path.exists(path) :
        os.makedirs(path)

    for j in range(data.shape[2]) : 
        cv2.imwrite(f'{path}/{j + 1}.png', data[:, :, j].T)
        print(f'Write file: {path}/{j + 1}.png')


def Write3DArrayAsNIFTI(path : str, data : np.ndarray) :
    nib_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nib_img, path)