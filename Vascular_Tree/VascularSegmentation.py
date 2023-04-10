import numpy as np
import cupy as cp
import cv2
from Utils import *
import FileManager

USE_HESSIAN_CACHE = True
BRONCHUS_DILATE_RADIUS = 5
MIN_INCOUNT = 100
MIN_FORK = 4
MAX_VOLUME_RATIO = 0.017
ROOT_THRESHOLD = 100
MAX_REPAIR_LOOP = 20
REPAIR_RADIUS = 3

def PrepareMask(lung_mask : np.ndarray, bronchus_tree : np.ndarray) -> np.ndarray :
    side_len = BRONCHUS_DILATE_RADIUS
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(side_len, side_len))

    mask = np.empty(bronchus_tree.shape, bronchus_tree.dtype)
    iters = bronchus_tree.shape[2] // 512
    for i in range(iters) :
        mask[:, :, i * 512 : (i+1) * 512] = cv2.dilate(bronchus_tree[:, :, i * 512 : (i+1) * 512], SE)
    mask[:, :, iters * 512:] = cv2.dilate(bronchus_tree[:, :, iters * 512:], SE)
    mask = np.logical_and(lung_mask, np.logical_not(mask))
    mask = mask.astype(np.uint8)
    mask *= 255
    return mask


def MedianFilter(data : np.ndarray) -> np.ndarray :
    result = np.empty(data.shape, data.dtype)
    iters = data.shape[2] // 512
    for i in range(iters) :
        result[:, :, i * 512 : (i+1) * 512] = cv2.medianBlur(data[:, :, i * 512 : (i+1) * 512], 3)
    result[:, :, iters * 512:] = cv2.medianBlur(data[:, :, iters * 512:], 3)
    return result


def Gauss3D(x, y, z, s : float) :
    return np.exp(-(x**2 + y**2 + z**2) / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)**3


def Hessian3D(data : np.ndarray, s : float) -> np.ndarray : 
    data = data.astype(np.float32)
    a = np.round(3 * s)
    X, Y, Z = np.mgrid[-a:a+1, -a:a+1, -a:a+1].astype(np.float32)

    G = Gauss3D(X, Y, Z, s)
    G *= s  # normalize
    G /= s**4

    Lxx = Convolve3d(data, (X**2 - s**2) * G)
    Lyy = Convolve3d(data, (Y**2 - s**2) * G)
    Lzz = Convolve3d(data, (Z**2 - s**2) * G)
    Lxy = Convolve3d(data, X * Y * G)
    Lxz = Convolve3d(data, X * Z * G)
    Lyz = Convolve3d(data, Y * Z * G)

    H = np.array([[Lxx, Lxy, Lxz],
                  [Lxy, Lyy, Lyz],
                  [Lxz, Lyz, Lzz]])
    return H


def HessianEnhance(data : np.ndarray, s : float) -> np.ndarray :
    a, b, c = (0.5, 0.5, 140.0)

    hessian = Hessian3D(data, s)
    result = np.empty(data.shape, dtype=np.float32)
    if USE_GPU :
        hessian = cp.asarray(hessian)
        e_vals = Eigvalssym3_GPU(hessian)
        del hessian
    
        Ra_2 = cp.square(cp.divide(e_vals[1], e_vals[2]))
        Rb_2 = cp.divide(cp.square(e_vals[0]), cp.abs(cp.multiply(e_vals[1], e_vals[2])))
        S_2 = cp.add(cp.add(cp.square(e_vals[0]), cp.square(e_vals[1])), cp.square(e_vals[2]))

        result = cp.multiply(cp.multiply(cp.subtract(1, cp.exp(cp.divide(cp.negative(Ra_2), 2 * a**2))), 
                                         cp.exp(cp.divide(cp.negative(Rb_2), 2 * b**2))), 
                             cp.subtract(1, cp.exp(cp.divide(cp.negative(S_2), 2 * c**2))))
        result[cp.logical_or(cp.greater_equal(e_vals[1], 0), cp.greater_equal(e_vals[2], 0))] = 0
        result = cp.asnumpy(result)

        del e_vals
        del Ra_2
        del Rb_2
        del S_2
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    else :
        e_vals = Eigvalssym3(hessian)
    
        Ra_2 = (e_vals[1] / e_vals[2])**2
        Rb_2 = e_vals[0]**2 / np.abs(e_vals[1] * e_vals[2])
        S_2 = e_vals[0]**2 + e_vals[1]**2 + e_vals[2]**2

        result = (1 - np.exp(-Ra_2 / (2 * a**2))) * np.exp(-Rb_2 / (2 * b**2)) * (1 - np.exp(-S_2 / (2 * c**2)))
        result[np.logical_or(e_vals[1] >= 0, e_vals[2] >= 0)] = 0

    return result


def VascularTreeExtraction(data : np.ndarray, 
                           mask : np.ndarray, 
                           raw_mask : np.ndarray,
                           cache_path: str) -> np.ndarray :
    scales = (1.0, 1.5, 2.0, 2.5, 3.0)
    thres = (100, 200, 250, 500, 700)
    enhanced_map = np.zeros(data.shape, dtype = bool)
    enhanced_data = np.zeros(data.shape, dtype = np.float32)

    # hessian enhanced
    for k in range(len(scales)) :
        s = scales[k]
        v = np.empty(data.shape, dtype = np.float32)
        if USE_HESSIAN_CACHE :
            v = FileManager.ReadPNGDir(cache_path + f'/_hessian_{s}', (data.shape[0], data.shape[1]))
            v = v.astype(np.float32)
            v /= 255
        else :
            v = HessianEnhance(data, s)
            FileManager.Write3DArrayAsImages(cache_path + f'/_hessian_{s}', v * 255)

        v *= 3071
        T = thres[k]
        
        iters = v.shape[2] // 512
        v_binary = np.empty(data.shape, dtype=bool)
        for i in range(iters) :
            ret, v_binary[:, :, i * 512 : (i+1) * 512] = cv2.threshold(v[:, :, i * 512 : (i+1) * 512], T, 1, cv2.THRESH_BINARY)
        ret, v_binary[:, :, iters * 512:] = cv2.threshold(v[:, :, iters * 512:], T, 1, cv2.THRESH_BINARY)

        enhanced_map = np.logical_or(enhanced_map, v_binary)
        enhanced_data = np.maximum(enhanced_data, v)

    # 3D connectivity
    enhanced_data[mask == 0] = 0
    enhanced_data[enhanced_map == 0] = 0
    enhanced_data[data < ROOT_THRESHOLD] = 0
    enhanced_map = np.logical_and(enhanced_map, mask)

    kernel = np.ones((3, 3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    result = np.zeros(data.shape, dtype = bool)
    while np.sum(enhanced_data > 0) >= MIN_INCOUNT :
        seed = np.argmax(enhanced_data)
        visit = np.zeros((data.size,), dtype = bool)
        visit[seed] = True
        visit = visit.reshape(data.shape)
        last_sum = 0

        if USE_GPU :
            visit = cp.asarray(visit)
            enhanced_map = cp.asarray(enhanced_map)
            curr_sum = cp.sum(visit)

            while curr_sum != last_sum :
                last_sum = curr_sum

                shift = cp.pad(visit, pad_width, mode = "edge")
                shift = cp.roll(shift, anchor, axis=(0, 1, 2))
                for x in range(kernel.shape[0]) :
                    for y in range(kernel.shape[1]) :
                        for z in range(kernel.shape[2]) :
                            if kernel[x, y, z] :
                                visit = cp.logical_or(visit, cp.logical_and(
                                    shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                    enhanced_map))
                            shift = cp.roll(shift, -1, axis=2)
                        shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                    shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                curr_sum = cp.sum(visit)

            visit = cp.asnumpy(visit)
            enhanced_map = cp.asnumpy(enhanced_map)
            del shift
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
        else:
            curr_sum = np.sum(visit)

            while curr_sum != last_sum :
                last_sum = curr_sum

                shift = np.pad(visit, pad_width, mode = "edge")
                shift = np.roll(shift, anchor, axis=(0, 1, 2))
                for x in range(kernel.shape[0]) :
                    for y in range(kernel.shape[1]) :
                        for z in range(kernel.shape[2]) :
                            if kernel[x, y, z] :
                                visit = np.logical_or(visit, np.logical_and(
                                    shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                    enhanced_map))
                            shift = np.roll(shift, -1, axis=2)
                        shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                    shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                curr_sum = np.sum(visit)
            
        enhanced_data[visit != 0] = 0

        # check region
        incount = np.sum(np.logical_and(visit, raw_mask))
        if incount >= MIN_INCOUNT :
            z = 0
            while not np.any(visit[:, :, z]) :
                z += 1
            z_min = z

            fork_num = 0
            last_num = 1
            curr_num = 1
            while z < visit.shape[2] :
                last_num = curr_num
                curr_num, labels = cv2.connectedComponents(visit[:, :, z].astype(np.uint8))
                if curr_num != last_num :
                    fork_num += 1
                if curr_num == 1 :
                    break
                z += 1
            z_max = z

            if fork_num >= MIN_FORK :
                x = 0
                while not np.any(visit[x, :, z_min : z_max]) :
                    x += 1
                x_min = x
                while np.any(visit[x, :, z_min : z_max]) :
                    x += 1
                x_max = x

                y = 0
                while not np.any(visit[x_min : x_max, y, z_min : z_max]) :
                    y += 1
                y_min = y
                while np.any(visit[x_min : x_max, y, z_min : z_max]) :
                    y += 1
                y_max = y

                vol_rect = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
                vol_ratio = curr_sum / vol_rect

                if vol_ratio <= MAX_VOLUME_RATIO :
                    result = np.logical_or(result, visit)

    # root expansion
    kernel = np.ones((3, 3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])
    
    last_sum = 0
    if USE_GPU :
        result = cp.asarray(result)
        data = cp.asarray(data)
        curr_sum = cp.sum(result)

        loop_count = 0
        while curr_sum != last_sum and loop_count < MAX_REPAIR_LOOP :
            last_sum = curr_sum

            shift = cp.pad(result, pad_width, mode = "edge")
            shift = cp.roll(shift, anchor, axis=(0, 1, 2))
            for x in range(kernel.shape[0]) :
                for y in range(kernel.shape[1]) :
                    for z in range(kernel.shape[2]) :
                        if kernel[x, y, z] :
                            result = cp.logical_or(result, cp.logical_and(
                                shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                cp.greater_equal(data, ROOT_THRESHOLD)))
                        shift = cp.roll(shift, -1, axis=2)
                    shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

            curr_sum = cp.sum(result)
            loop_count += 1

        result = cp.asnumpy(result)
        data = cp.asnumpy(data)
        del shift
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    else :
        sum = np.sum(result)

        loop_count = 0
        while curr_sum != last_sum and loop_count < MAX_REPAIR_LOOP :
            last_sum = curr_sum

            shift = np.pad(result, pad_width, mode = "edge")
            shift = np.roll(shift, anchor, axis=(0, 1, 2))
            for x in range(kernel.shape[0]) :
                for y in range(kernel.shape[1]) :
                    for z in range(kernel.shape[2]) :
                        if kernel[x, y, z] :
                            result = np.logical_or(result, np.logical_and(
                                shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                data >= ROOT_THRESHOLD))
                        shift = np.roll(shift, -1, axis=2)
                    shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

            curr_sum = np.sum(result)
            loop_count += 1

    result = result.astype(np.uint8)
    result *= 255
    return result


def MorphologicalRepair(data : np.ndarray) -> np.ndarray :
    side_len = REPAIR_RADIUS
    SE = cv2.getStructuringElement(cv2.MORPH_RECT,(side_len, side_len))
    
    # close operation
    result = np.empty(data.shape, data.dtype)
    iters = data.shape[2] // 512
    for i in range(iters) :
        result[:, :, i * 512 : (i+1) * 512] = cv2.morphologyEx(data[:, :, i * 512 : (i+1) * 512], cv2.MORPH_CLOSE, SE)
    result[:, :, iters * 512:] = cv2.morphologyEx(data[:, :, iters * 512:], cv2.MORPH_CLOSE, SE)

    return result


def Execute(data : np.ndarray, 
            raw_mask : np.ndarray, 
            lung_mask : np.ndarray, 
            bronchus_tree : np.ndarray,
            cache_path: str) -> np.ndarray :
    mask = PrepareMask(lung_mask, bronchus_tree)
    processed_data = MedianFilter(data)
    result = VascularTreeExtraction(processed_data, mask, raw_mask, cache_path)
    result = MorphologicalRepair(result)
    return result