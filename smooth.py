# coding=utf-8
import cv2
import numpy as np


def getWindow(img, i, j, win_size):
    # 获得指定范围、大小的窗口内容
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2
    start_x = i - half_size
    start_y = j - half_size
    end_x = i + half_size + 1
    end_y = j + half_size + 1
    win = img[start_x:end_x, start_y:end_y]
    return win


def smoothImg(img, canny_th1=128, canny_th2=255, kernel_size=5, dilate_iter=1, flag=1, method=1, win_size=5,
              win_offset=1):
    """

    :param img: 输入影像，单波段或RGB
    :param canny_th1: Canny算子低阈值，默认128
    :param canny_th2: Canny算子高阈值，默认255
    :param kernel_size: 膨胀运算卷积核大小，默认为5
    :param dilate_iter: 膨胀迭代次数，默认为1
    :param flag: 是否开启边缘强化平滑，1-开启，0-关闭，默认为1
    :param method: 平滑方法，1-均值滤波，2-中值滤波，默认为1
    :param win_size: 平滑窗口大小，默认为5
    :param win_offset: 平滑窗口移动步长，默认为1
    :return: 平滑后的影像
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if len(img.shape) == 2:
        # single band
        edges = cv2.Canny(img, canny_th1, canny_th2)
        mask = cv2.dilate(edges, kernel, iterations=dilate_iter)
        cv2.imwrite("mask.jpg", mask)
        img_smooth = smooth_core(img, mask, flag=flag, method=method, win_size=win_size, win_offset=win_offset)
        return img_smooth
    else:
        # BGR band
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, canny_th1, canny_th2)
        mask = cv2.dilate(edges, kernel, iterations=dilate_iter)
        cv2.imwrite("mask.jpg", mask)
        band_b = img[:, :, 0]
        band_g = img[:, :, 1]
        band_r = img[:, :, 2]
        print 'processing band blue...'
        band_b_s = smooth_core(band_b, mask, flag=flag, method=method, win_size=win_size, win_offset=win_offset)
        print 'processing band green...'
        band_g_s = smooth_core(band_g, mask, flag=flag, method=method, win_size=win_size, win_offset=win_offset)
        print 'processing band red...'
        band_r_s = smooth_core(band_r, mask, flag=flag, method=method, win_size=win_size, win_offset=win_offset)
        img_smooth = np.zeros_like(img)
        img_smooth[:, :, 0] = band_b_s
        img_smooth[:, :, 1] = band_g_s
        img_smooth[:, :, 2] = band_r_s
        return img_smooth


def smooth_core(img, mask, flag=1, method=1, win_size=5, win_offset=1):
    img_smooth = np.zeros_like(img)
    if method == 1:
        bk = cv2.blur(img, (win_size, win_size))
    elif method == 2:
        bk = cv2.medianBlur(img, (win_size, win_size))
    img_smooth[:, :] = bk
    width = img.shape[1]
    height = img.shape[0]
    safe_range = win_offset + win_size
    if flag == 1:
        for i in range(safe_range, height - safe_range):
            if i % 50 == 0:
                print i, '/', height - safe_range
            for j in range(safe_range, width - safe_range):
                val = mask[i, j]
                if val == 255:
                    img_smooth[i, j] = img[i, j]
                else:
                    win = getWindow(img, i, j, win_size)
                    if method == 1:
                        win_mean = np.mean(win)
                    elif method == 2:
                        win_mean = np.median(win)
                    img_smooth[i, j] = win_mean
    else:
        for i in range(safe_range, height - safe_range):
            if i % 50 == 0:
                print i, '/', height - safe_range
            for j in range(safe_range, width - safe_range):
                win = getWindow(img, i, j, win_size)
                if method == 1:
                    win_mean = np.mean(win)
                elif method == 2:
                    win_mean = np.median(win)
                img_smooth[i, j] = win_mean
    return img_smooth


if __name__ == '__main__':
    img = cv2.imread("IMG_0284.JPG")

    img_s = smoothImg(img,
                      canny_th1=20, canny_th2=255,
                      kernel_size=5, dilate_iter=1,
                      flag=1, method=1,
                      win_size=5, win_offset=1)
    cv2.imwrite("smoothWithEgde.jpg", img_s)

    img_bi = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite("compare_bilateral.jpg", img_bi)

    img_blur = cv2.blur(img, (5, 5))
    cv2.imwrite("compare_blur.jpg", img_blur)

    img_median = cv2.medianBlur(img, 5)
    cv2.imwrite("compare_median.jpg", img_median)

    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("compare_gauss.jpg", img_gauss)
