import numpy as np
import cupy as cp
import cv2
from threading import Thread

USE_GPU = True
PARALLELIZATION_LEVEL = 0

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

class ResThread(Thread) :
    def __init__(self, func, args) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


def Convolve3d_CPU(data : np.ndarray, kernel : np.ndarray) -> np.ndarray :
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    result = np.zeros(data.shape, dtype = data.dtype)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    shift = np.pad(data, pad_width, mode = "edge")
    shift = np.roll(shift, anchor, axis=(0, 1, 2))

    for x in range(kernel.shape[0]) :
        for y in range(kernel.shape[1]) :
            for z in range(kernel.shape[2]) :
                result += shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width] * kernel[x, y, z]
                shift = np.roll(shift, -1, axis=2)

            shift = np.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))

        shift = np.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

    return result


def Convolve3d_GPU(data : cp.ndarray, kernel : cp.ndarray) -> cp.ndarray :
    anchor = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
    result = cp.zeros(data.shape, dtype = data.dtype)
    pad_width = np.maximum(np.maximum(anchor[0], anchor[1]), anchor[2])

    shift = cp.pad(data, pad_width, mode = "edge")
    shift = cp.roll(shift, anchor, axis=(0, 1, 2))

    for x in range(kernel.shape[0]) :
        for y in range(kernel.shape[1]) :
            for z in range(kernel.shape[2]) :
                result = cp.add(result, cp.multiply(
                    shift[pad_width : -pad_width, pad_width : -pad_width, pad_width : -pad_width], 
                    kernel[x, y, z]))
                shift = cp.roll(shift, -1, axis=2)

            shift = cp.roll(shift, (-1, kernel.shape[2]), axis=(1, 2))

        shift = cp.roll(shift, (-1, kernel.shape[1]), axis=(0, 1))

    del shift
    return result


def Convolve3d(data : np.ndarray, kernel : np.ndarray) -> np.ndarray :
    if USE_GPU :
        cp_data = cp.asarray(data)
        cp_kernel = cp.asarray(kernel)
        cp_result = Convolve3d_GPU(cp_data, cp_kernel)
        result = cp.asnumpy(cp_result)
        
        del cp_data
        del cp_kernel
        del cp_result
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return result
    else :
        result = Convolve3d_CPU(data, kernel) 
        return result


def Eigvalssym3_CPU(mat : np.ndarray) -> np.ndarray :
    b = -(mat[0, 0] + mat[1, 1] + mat[2, 2])
    c = (mat[0, 0] * (mat[1, 1] + mat[2, 2]) + mat[1, 1] * mat[2, 2] 
         - mat[0, 1]**2 - mat[0, 2]**2 - mat[1, 2]**2)
    d = (mat[0, 0] * mat[1, 2]**2 + mat[1, 1] * mat[0, 2]**2 + mat[2, 2] * mat[0, 1]**2 
         + 2 * mat[0, 1] * mat[0, 2] * mat[1, 2] - mat[0, 0] * mat[1, 1] * mat[2, 2])

    b_div_3 = b / 3
    p = c - 3 * b_div_3**2
    q = d - b_div_3 * c + 2 * b_div_3**3
    del b
    del c
    del d

    nq_div_2 = q / 2
    delta = nq_div_2**2 + (p / 3)**3
    sqrt_delta = np.sqrt(delta.astype(np.complex64))
    cbrt_nq2_add_sdel = np.power(nq_div_2 + sqrt_delta, 1/3)
    cbrt_nq2_sub_sdel = np.power(nq_div_2 - sqrt_delta, 1/3)
    del p
    del q
    del nq_div_2
    del delta
    del sqrt_delta

    w = -1/2 + np.sqrt(3)/2 * 1j
    w = w.astype(np.complex64)
    w_2 = -1/2 - np.sqrt(3)/2 * 1j
    w_2 = w_2.astype(np.complex64)

    y1 = cbrt_nq2_add_sdel + cbrt_nq2_sub_sdel
    y2 = w * cbrt_nq2_add_sdel + w_2 * cbrt_nq2_sub_sdel
    y3 = w_2 * cbrt_nq2_add_sdel + w * cbrt_nq2_sub_sdel
    del cbrt_nq2_add_sdel
    del cbrt_nq2_sub_sdel

    x1 = y1 - b_div_3
    x2 = y2 - b_div_3
    x3 = y3 - b_div_3
    del b_div_3
    del y1
    del y2
    del y3
    
    e_vals = np.array([np.real(x1), np.real(x2), np.real(x3)])
    abs_vals = np.abs(e_vals)
    del x1
    del x2
    del x3

    # sort
    min0 = np.logical_and(abs_vals[0] <= abs_vals[1], abs_vals[0] <= abs_vals[2])
    min1 = np.logical_and(np.logical_not(min0), abs_vals[1] <= abs_vals[2])
    min2 = np.logical_and(np.logical_not(min0), np.logical_not(min1))
    max0 = np.logical_and(abs_vals[0] > abs_vals[1], abs_vals[0] > abs_vals[2])
    max1 = np.logical_and(np.logical_not(max0), abs_vals[1] > abs_vals[2])
    max2 = np.logical_and(np.logical_not(max0), np.logical_not(max1))
    mid0 = np.logical_and(np.logical_not(min0), np.logical_not(max0))
    mid1 = np.logical_and(np.logical_not(min1), np.logical_not(max1))
    mid2 = np.logical_and(np.logical_not(min2), np.logical_not(max2))

    e1 = min0 * e_vals[0] + min1 * e_vals[1] + min2 * e_vals[2]
    e2 = mid0 * e_vals[0] + mid1 * e_vals[1] + mid2 * e_vals[2]
    e3 = max0 * e_vals[0] + max1 * e_vals[1] + max2 * e_vals[2]
    del min0
    del min1
    del min2
    del mid0
    del mid1
    del mid2
    del max0
    del max1
    del max2

    return np.array([e1, e2, e3])


def Eigvalssym3_GPU(mat : cp.ndarray) -> cp.ndarray :
    b = cp.negative(cp.add(cp.add(mat[0, 0], mat[1, 1]), mat[2, 2]))
    c = cp.subtract(cp.add(cp.multiply(mat[0, 0], cp.add(mat[1, 1], mat[2, 2])), cp.multiply(mat[1, 1], mat[2, 2])), 
                    cp.add(cp.add(cp.square(mat[0, 1]), cp.square(mat[0, 2])), cp.square(mat[1, 2])))
    d = cp.add(cp.add(cp.add(cp.multiply(mat[0, 0], cp.square(mat[1, 2])), cp.multiply(mat[1, 1], cp.square(mat[0, 2]))),
                      cp.multiply(mat[2, 2], cp.square(mat[0, 1]))),
               cp.subtract(cp.multiply(cp.multiply(2, mat[0, 1]), cp.multiply(mat[0, 2], mat[1, 2])),
                           cp.multiply(cp.multiply(mat[0, 0], mat[1, 1]), mat[2, 2])))

    b_div_3 = cp.divide(b, 3)
    p = cp.subtract(c, cp.multiply(3, cp.square(b_div_3)))
    q = cp.add(cp.subtract(d, cp.multiply(b_div_3, c)), cp.multiply(2, cp.power(b_div_3, 3)))
    del b
    del c
    del d

    nq_div_2 = cp.negative(cp.divide(q, 2))
    delta = cp.add(cp.square(nq_div_2), cp.power(cp.divide(p, 3), 3))
    sqrt_delta = cp.sqrt(delta.astype(cp.complex64))
    cbrt_nq2_add_sdel = cp.power(cp.add(nq_div_2, sqrt_delta), 1/3)
    cbrt_nq2_sub_sdel = cp.power(cp.subtract(nq_div_2, sqrt_delta), 1/3)
    del p
    del q
    del nq_div_2
    del delta
    del sqrt_delta

    w = -1/2 + cp.sqrt(3)/2 * 1j
    w = w.astype(cp.complex64)
    w_2 = -1/2 - cp.sqrt(3)/2 * 1j
    w_2 = w_2.astype(cp.complex64)

    y1 = cp.add(cp.real(cbrt_nq2_add_sdel), cp.real(cbrt_nq2_sub_sdel))
    y2 = cp.add(cp.real(cp.multiply(w, cbrt_nq2_add_sdel)), cp.real(cp.multiply(w_2, cbrt_nq2_sub_sdel)))
    y3 = cp.add(cp.real(cp.multiply(w_2, cbrt_nq2_add_sdel)), cp.real(cp.multiply(w, cbrt_nq2_sub_sdel)))
    del cbrt_nq2_add_sdel
    del cbrt_nq2_sub_sdel

    x1 = cp.subtract(y1, b_div_3)
    x2 = cp.subtract(y2, b_div_3)
    x3 = cp.subtract(y3, b_div_3)
    del b_div_3
    del y1
    del y2
    del y3
    
    e_vals = cp.array([x1, x2, x3])
    abs_vals = cp.abs(e_vals)
    del x1
    del x2
    del x3

    # sort
    min0 = cp.logical_and(cp.less_equal(abs_vals[0], abs_vals[1]), cp.less_equal(abs_vals[0], abs_vals[2]))
    min1 = cp.logical_and(cp.logical_not(min0), cp.less_equal(abs_vals[1], abs_vals[2]))
    min2 = cp.logical_and(cp.logical_not(min0), cp.logical_not(min1))
    max0 = cp.logical_and(cp.greater(abs_vals[0], abs_vals[1]), cp.greater(abs_vals[0], abs_vals[2]))
    max1 = cp.logical_and(cp.logical_not(max0), cp.greater(abs_vals[1], abs_vals[2]))
    max2 = cp.logical_and(cp.logical_not(max0), cp.logical_not(max1))
    mid0 = cp.logical_and(cp.logical_not(min0), cp.logical_not(max0))
    mid1 = cp.logical_and(cp.logical_not(min1), cp.logical_not(max1))
    mid2 = cp.logical_and(cp.logical_not(min2), cp.logical_not(max2))
    del abs_vals

    e1 = cp.empty(e_vals[0].shape, cp.float32)
    e1[min0] = e_vals[0][min0]
    e1[min1] = e_vals[1][min1]
    e1[min2] = e_vals[2][min2]
    del min0
    del min1
    del min2
    e2 = cp.empty(e_vals[0].shape, cp.float32)
    e2[mid0] = e_vals[0][mid0]
    e2[mid1] = e_vals[1][mid1]
    e2[mid2] = e_vals[2][mid2]
    del mid0
    del mid1
    del mid2
    e3 = cp.empty(e_vals[0].shape, cp.float32)
    e3[max0] = e_vals[0][max0]
    e3[max1] = e_vals[1][max1]
    e3[max2] = e_vals[2][max2]
    del max0
    del max1
    del max2

    ret = cp.array([e1, e2, e3])
    del e1
    del e2
    del e3

    return ret


def Eigvalssym3(mat : np.ndarray) -> np.ndarray :
    if USE_GPU :
        cp_mat = cp.asarray(mat)
        cp_result = Eigvalssym3_GPU(cp_mat)
        result = cp.asnumpy(cp_result)

        del cp_mat
        del cp_result
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return result
    else :
        result = Eigvalssym3_CPU(mat)
        return result