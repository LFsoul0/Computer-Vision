import numpy as np
import cupy as cp
import cv2
from Utils import *

CT_MIN = -950
CT_MAX = -775
TOP_RANGE = (-15, -10)
GROWTH_STEP = 0.02
MIN_PIXELS = 20000
LEAK_THRESHOLD = 0.1
REPAIR_RADIUS = 5

def Normalization(data : np.ndarray) -> np.ndarray :
    data = data.astype(np.float32)
    result = (data - CT_MIN) / (CT_MAX - CT_MIN)
    result[result < 0] = 0
    result[result > 1] = 1
    return result


def MeanFilter(data : np.ndarray) -> np.ndarray :
    result = np.empty(data.shape, data.dtype)
    iters = data.shape[2] // 512
    for i in range(iters) :
        result[:, :, i * 512 : (i+1) * 512] = cv2.blur(data[:, :, i * 512 : (i+1) * 512], (3, 3))
    result[:, :, iters * 512:] = cv2.blur(data[:, :, iters * 512:], (3, 3))

    left_shift = result.copy()
    left_shift[:, :, : -1] = left_shift[:, :, 1 : ]
    right_shift = result.copy()
    right_shift[:, :, 1 : ] = right_shift[:, :, : -1]
    result = result / 3 + left_shift / 3 + right_shift / 3

    return result


def Gradient(data : np.ndarray) -> np.ndarray :
    data = data.astype(np.int16)
    result = np.zeros(data.shape, dtype = np.float32)

    kernel = np.ones((3, 3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    if USE_GPU :
        data = cp.asarray(data)
        result = cp.asarray(result)

        shift = cp.pad(data, pad_width, mode = "edge")
        shift = cp.roll(shift, anchor, axis=(0, 1, 2))
        for x in range(kernel.shape[0]) :
            for y in range(kernel.shape[1]) :
                for z in range(kernel.shape[2]) :
                    if kernel[x, y, z] :
                        result = cp.maximum(result, cp.abs(cp.subtract(
                            shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width],
                            data)))
                    shift = cp.roll(shift, -1, axis=2)
                shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
            shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

        data = cp.asnumpy(data)
        result = cp.asnumpy(result)
        del shift
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    else :
        shift = np.pad(data, pad_width, mode = "edge")
        shift = np.roll(shift, anchor, axis=(0, 1, 2))
        for x in range(kernel.shape[0]) :
            for y in range(kernel.shape[1]) :
                for z in range(kernel.shape[2]) :
                    if kernel[x, y, z] :
                        gradient = np.abs(
                            shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width] -
                            data)
                        result = np.maximum(result, gradient)
                    shift = np.roll(shift, -1, axis=2)
                shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
            shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

    result /= CT_MAX - CT_MIN
    result[result > 1] = 1

    return result


def BronchusTreeExtraction(f : np.ndarray, lung_mask : np.ndarray) -> np.ndarray :
    # select seed 
    f_top = f[:, :, TOP_RANGE[0] : TOP_RANGE[1]]
    mask_top = lung_mask[:, :, TOP_RANGE[0] : TOP_RANGE[1]]
    if np.sum(mask_top != 0) == 0 :
        #print(f'Bronchus tree extraction error: no valid seed')
        raise Exception('Bronchus tree extraction error: no valid seed')
        return np.zeros(f.shape, dtype = bool)

    f_top[mask_top == 0] = 1
    prime_t = np.min(f_top)
    candidates = np.where(f_top <= prime_t)
    seed = (candidates[0][0], candidates[1][0], f.shape[2] + TOP_RANGE[0] + candidates[2][0])

    # 3D connective growth
    kernel = np.ones((3, 3, 3), dtype = bool)
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    result = np.zeros(f.shape, dtype = bool)
    result[seed] = 1
    sum = 1
    next_sum = 1
    T = prime_t

    if USE_GPU:
        f = cp.asarray(f)
        result = cp.asarray(result)
        next_result = result.copy()

        while T <= prime_t + GROWTH_STEP or sum < MIN_PIXELS or (next_sum - sum) / next_sum <= LEAK_THRESHOLD :
            result = next_result.copy()
            sum = next_sum

            last_sum = 0
            while next_sum != last_sum :
                last_sum = next_sum

                shift = cp.pad(next_result, pad_width, mode = "edge")
                shift = cp.roll(shift, anchor, axis=(0, 1, 2))
                for x in range(kernel.shape[0]) :
                    for y in range(kernel.shape[1]) :
                        for z in range(kernel.shape[2]) :
                            if kernel[x, y, z] :
                                next_result = cp.logical_or(next_result, cp.logical_and(
                                    shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                    cp.less_equal(f, T)))
                            shift = cp.roll(shift, -1, axis=2)
                        shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                    shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                next_sum = cp.sum(next_result)

            T += GROWTH_STEP
            if T >= 1 :
                #print(f'Bronchus tree extraction error: threshold overflow')
                raise Exception('Bronchus tree extraction error: threshold overflow')
                break

        f = cp.asnumpy(f)
        result = cp.asnumpy(result)
        del next_result
        del shift
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    else :
        next_result = result.copy()

        while T <= prime_t + GROWTH_STEP or sum < MIN_PIXELS or (next_sum - sum) / next_sum <= LEAK_THRESHOLD :
            if T >= 1 :
                break

            result = next_result.copy()
            sum = next_sum

            last_sum = 0
            while next_sum != last_sum :
                last_sum = next_sum

                shift = np.pad(next_result, pad_width, mode = "edge")
                shift = np.roll(shift, anchor, axis=(0, 1, 2))
                for x in range(kernel.shape[0]) :
                    for y in range(kernel.shape[1]) :
                        for z in range(kernel.shape[2]) :
                            if kernel[x, y, z] :
                                next_result = np.logical_or(next_result, np.logical_and(
                                    shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                                    f <= T))
                            shift = np.roll(shift, -1, axis=2)
                        shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))
                    shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

                next_sum = np.sum(next_result)

            T += GROWTH_STEP

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


def Execute(data : np.ndarray, lung_mask : np.ndarray) -> np.ndarray :
    a, b, c = (0.35, 0.4, 0.25)

    data = data.astype(np.float32)

    f1 = Normalization(data)
    f2 = MeanFilter(data)
    f2 = Normalization(f2)
    f3 = Gradient(data)
    f = a * f1 + b * f2 + c * f3
    
    result = BronchusTreeExtraction(f, lung_mask)
    result = MorphologicalRepair(result)

    return result