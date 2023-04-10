from typing import Tuple
import numpy as np
import cupy as cp
import cv2
from Utils import *

PRIME_THRESHOLD = -500
REPAIR_RADIUS = 17


def Preprocessing(data : np.ndarray) -> np.ndarray :
    iters = data.shape[2] // 512
    result = np.empty(data.shape, data.dtype)
    for i in range(iters) :
        result[:, :, i * 512 : (i+1) * 512] = cv2.medianBlur(data[:, :, i * 512 : (i+1) * 512], 3)
        result[:, :, i * 512 : (i+1) * 512] = cv2.GaussianBlur(data[:, :, i * 512 : (i+1) * 512], (7, 7), 1)

    result[:, :, iters * 512:] = cv2.medianBlur(data[:, :, iters * 512:], 3)
    result[:, :, iters * 512:] = cv2.GaussianBlur(data[:, :, iters * 512:], (7, 7), 1)
    return result

def Binarization(data : np.ndarray, inv : bool = True) -> np.ndarray :
    mode = cv2.THRESH_BINARY_INV
    if not inv :
        cv2.THRESH_BINARY
       
    result = np.empty(data.shape, data.dtype)
    iters = data.shape[2] // 512
    for i in range(iters) :
        ret, result[:, :, i * 512 : (i+1) * 512] = cv2.threshold(data[:, :, i * 512 : (i+1) * 512], PRIME_THRESHOLD, 255, mode)
    ret, result[:, :, iters * 512:] = cv2.threshold(data[:, :, iters * 512:], PRIME_THRESHOLD, 255, mode)
    return result

def LungParenchymaExtraction(data : np.ndarray) -> np.ndarray :
    bool_data = np.zeros(data.shape, dtype = bool)
    bool_data[data != 0] = 1
    data = bool_data

    # select seed 
    seed_left = (0, 0, 0)
    seed_right = (0, 0, 0)

    kernel = np.ones((3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    pad_width = np.maximum(anchor[0], anchor[1])

    for z in range(data.shape[2] // 2 - 5, data.shape[2] // 2 + 5) :
        # left side
        visit = np.zeros((data.shape[0], data.shape[1]), dtype = bool)
        for y in range(data.shape[1] // 2 - 10, data.shape[1] // 2 + 10) :
            for x in range(data.shape[0] // 2 - 150, data.shape[0] // 2) : 
                if visit[x, y] or data[x, y, z] == 0 :
                    continue

                if USE_GPU :
                    visit = cp.asarray(visit)
                    data_layer = cp.asarray(data[:, :, z])

                    prev_count = cp.sum(visit)
                    last_count = prev_count
                    visit[x, y] = True
                    count = cp.sum(visit)

                    while count != last_count :
                        last_count = count

                        shift = cp.pad(visit, pad_width, mode = "edge")
                        shift = cp.roll(shift, anchor, axis=(0, 1))
                        for i in range(kernel.shape[0]) :
                            for j in range(kernel.shape[1]) :
                                if kernel[i, j] :
                                    visit = cp.logical_or(visit, cp.logical_and(
                                        shift[pad_width : -pad_width, pad_width : -pad_width], 
                                        data_layer))
                                shift = cp.roll(shift, -1, axis=1)
                            shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                        count = cp.sum(visit)

                    visit = cp.asnumpy(visit)
                    del data_layer
                    del shift
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                else :
                    prev_count = np.sum(visit)
                    last_count = prev_count
                    visit[x, y] = True
                    count = np.sum(visit)

                    while count != last_count :
                        last_count = count

                        shift = np.pad(visit, pad_width, mode = "edge")
                        shift = np.roll(shift, anchor, axis=(0, 1))
                        for i in range(kernel.shape[0]) :
                            for j in range(kernel.shape[1]) :
                                if kernel[i, j] :
                                    visit = np.logical_or(visit, np.logical_and(
                                        shift[pad_width : -pad_width, pad_width : -pad_width], 
                                        data[:, :, z]))
                                shift = np.roll(shift, -1, axis=1)
                            shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                        count = np.sum(visit)

                if visit[0, -1] or visit[-1, -1] or visit[0, 0] or visit[-1, 0] :
                    visit = np.zeros((data.shape[0], data.shape[1]), dtype = bool)
                    continue

                if count - prev_count > 1000 :
                    seed_left = (x, y, z)
                    break

            if seed_left[2] == z :
                break

        # right side
        visit = np.zeros((data.shape[0], data.shape[1]), dtype = bool)
        for y in range(data.shape[1] // 2 - 10, data.shape[1] // 2 + 10) :
            for x in range(data.shape[0] // 2, data.shape[0] // 2 + 150) : 
                if visit[x, y] or data[x, y, z] == 0 :
                    continue

                if USE_GPU :
                    visit = cp.asarray(visit)
                    data_layer = cp.asarray(data[:, :, z])

                    prev_count = cp.sum(visit)
                    last_count = prev_count
                    visit[x, y] = True
                    count = cp.sum(visit)

                    while count != last_count :
                        last_count = count

                        shift = cp.pad(visit, pad_width, mode = "edge")
                        shift = cp.roll(shift, anchor, axis=(0, 1))
                        for i in range(kernel.shape[0]) :
                            for j in range(kernel.shape[1]) :
                                if kernel[i, j] :
                                    visit = cp.logical_or(visit, cp.logical_and(
                                        shift[pad_width : -pad_width, pad_width : -pad_width], 
                                        data_layer))
                                shift = cp.roll(shift, -1, axis=1)
                            shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                        count = cp.sum(visit)

                    visit = cp.asnumpy(visit)
                    del data_layer
                    del shift
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                else :
                    prev_count = np.sum(visit)
                    last_count = prev_count
                    visit[x, y] = True
                    count = np.sum(visit)

                    while count != last_count :
                        last_count = count

                        shift = np.pad(visit, pad_width, mode = "edge")
                        shift = np.roll(shift, anchor, axis=(0, 1))
                        for i in range(kernel.shape[0]) :
                            for j in range(kernel.shape[1]) :
                                if kernel[i, j] :
                                    visit = np.logical_or(visit, np.logical_and(
                                        shift[pad_width : -pad_width, pad_width : -pad_width], 
                                        data[:, :, z]))
                                shift = np.roll(shift, -1, axis=1)
                            shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                        count = np.sum(visit)
                        
                if visit[0, -1] or visit[-1, -1] or visit[0, 0] or visit[-1, 0] :
                    visit = np.zeros((data.shape[0], data.shape[1]), dtype = bool)
                    continue

                if count - prev_count > 1000 :
                    seed_right = (x, y, z)
                    break

            if seed_right[2] == z :
                break

        if seed_left[2] == z and seed_right[2] == z :
            break

    if seed_left[2] != seed_right[2] or seed_left[2] == 0:
        #print("Lung parenchyma extraction error: cannot find valid seed")
        raise Exception('Lung parenchyma extraction error: cannot find valid seed')
        return data

    # 3D connectivity
    result = np.zeros(data.shape, dtype = bool)
    kernel = np.ones((3, 3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    result[seed_left] = 1
    result[seed_right] = 1
    last_sum = 0

    if USE_GPU :
        result = cp.asarray(result)
        data = cp.asarray(data)
        sum = cp.sum(result)

        while sum != last_sum :
            last_sum = sum

            shift = cp.pad(result, pad_width, mode = "edge")
            shift = cp.roll(shift, anchor, axis=(0, 1, 2))
            for x in range(kernel.shape[0]) :
                for y in range(kernel.shape[1]) :
                    for z in range(kernel.shape[2]) :
                        if kernel[x, y, z] :
                            result = cp.logical_or(result, cp.logical_and(
                                shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                data))
                        shift = cp.roll(shift, -1, axis=2)
                    shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

            sum = cp.sum(result)

        result = cp.asnumpy(result)
        data = cp.asnumpy(data)
        del shift
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    else :
        sum = np.sum(result)

        while sum != last_sum :
            last_sum = sum

            shift = np.pad(result, pad_width, mode = "edge")
            shift = np.roll(shift, anchor, axis=(0, 1, 2))
            for x in range(kernel.shape[0]) :
                for y in range(kernel.shape[1]) :
                    for z in range(kernel.shape[2]) :
                        if kernel[x, y, z] :
                            result = np.logical_or(result, np.logical_and(
                                shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                data))
                        shift = np.roll(shift, -1, axis=2)
                    shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

            sum = np.sum(result)
    

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


def Execute(data : np.ndarray) -> Tuple[np.ndarray, np.ndarray] :
    processed_data = Preprocessing(data)
    binary_data = Binarization(processed_data)
    raw_mask = LungParenchymaExtraction(binary_data)
    lung_mask = MorphologicalRepair(raw_mask)
    return raw_mask, lung_mask