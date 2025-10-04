import cv2
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm


def merge_pairs(data: list, threshold: float) -> list:
    out = []
    i = 0
    n = len(data)
    while i < n:
        if (i + 1 < n) and (abs(data[i+1] - data[i]) < threshold):
            out.append(int((data[i] + data[i+1]) / 2))
            i += 2
        else:
            out.append(int(data[i]))
            i += 1
    return out


def get_boxes(h: list, v: list) -> list:
    rects = []
    for h_top, h_bottom in zip(h, h[1:]):
        for v_left, v_right in zip(v, v[1:]):
            rects.append([v_left, h_top, v_right - v_left, h_bottom - h_top])
    return rects


def split_boxes(boxes: list, bound: list) -> list:
    CORR_COEF = 5
    ret = []
    min_box = [float('inf'), float('inf'), float('inf'), float('inf')]
    for b in boxes:
        if min_box[2]*min_box[3] > b[2]*b[3]:
            min_box = b
    grid_cols = bound[2] // (min_box[2] - CORR_COEF)
    grid_rows = bound[3] // (min_box[3])
    grid = np.zeros((grid_rows, grid_cols, 4), dtype=np.int32)

    print(f"mat size {grid_rows}, {grid_cols} {len(boxes)=}")
    cnt = 0
    cnt_cols = 0
    cnt_rows = 0
    for box in boxes:
        while grid[cnt_rows][cnt_cols].any():
            cnt_cols = cnt % grid_cols
            cnt_rows = cnt // grid_cols
            cnt += 1

        wd = box[2]//(min_box[2] - CORR_COEF)
        hd = box[3]//(min_box[3] - CORR_COEF)
        x, y, w, h = box
        sub_w = w / wd
        sub_h = h / hd

        for i in range(hd):
            for j in range(wd):
                sub_x = x + j * sub_w
                sub_y = y + i * sub_h
                grid[cnt_rows + i][cnt_cols + j] = [int(sub_x), int(
                    sub_y), int(sub_w), int(sub_h)]
    return grid, grid_cols, grid_rows


def get_vec_from_pic(image):
    original = image.copy()
    horizontal_mask = np.zeros(image.shape, dtype=np.uint8)
    vertical_mask = np.zeros(image.shape, dtype=np.uint8)
    vertical_lines = list()
    horizontal_lines = list()
    bound_box = list()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape[:2]

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

    cnts = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bound_box = [0, 0, 0, 0]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        bound_box = [x, y, w, h] if bound_box[2] * \
            bound_box[3] < w * h else bound_box
        cv2.drawContours(thresh, [c], -1, (36, 255, 12), 2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    detect_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        horizontal_lines.append(cv2.fitLine(
            c, cv2.DIST_L2, 0, 0.01, 0.01)[3].tolist()[0])
        cv2.drawContours(horizontal_mask, [c], -1, (36, 255, 12), 2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    detect_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(
        detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        vertical_lines.append(cv2.fitLine(
            c, cv2.DIST_L2, 0, 0.01, 0.01)[2].tolist()[0])
        cv2.drawContours(vertical_mask, [c], -1, (36, 255, 12), 2)

    horizontal_lines = sorted(horizontal_lines)
    horizontal_lines = merge_pairs(horizontal_lines, 10)
    print(horizontal_lines)
    for n in horizontal_lines:
        cv2.line(original, (cols-1, int(n)),
                 (0, int(n)), (0, 0, 255), 2)

    vertical_lines = sorted(vertical_lines)
    vertical_lines = merge_pairs(vertical_lines, 10)
    print(vertical_lines)
    for n in vertical_lines:
        cv2.line(original, (int(n), rows-1),
                 (int(n), 0), (0, 0, 255), 2)

    boxes = get_boxes(horizontal_lines, vertical_lines)
    grid, grid_cals, grid_rows = split_boxes(boxes, bound_box)
    rast_vec = []
    THRESH_MAX = 127
    for row in grid:
        for col in row:
            x, y, w, h = col.tolist()
            tmp = np.mean(
                gray[y:y+h, x:x+w], axis=(0, 1))
            rast_vec.append(1 if tmp > THRESH_MAX else 0)

    return original, rast_vec, grid_cals, grid_rows


image_1 = cv2.imread('res/image_1.png')
image_1, vec_1, grid_cals, grid_rows = get_vec_from_pic(image_1)
print([vec_1[i:i + grid_cals] for i in range(0, len(vec_1), grid_cals)])

image_2 = cv2.imread('res/image_2.png')
image_2, vec_2, grid_cals, grid_rows = get_vec_from_pic(image_2)
print([vec_2[i:i + grid_cals] for i in range(0, len(vec_2), grid_cals)])

image_3 = cv2.imread('res/image_3.png')
image_3, vec_3, grid_cals, grid_rows = get_vec_from_pic(image_3)
print([vec_3[i:i + grid_cals] for i in range(0, len(vec_3), grid_cals)])

cos_sim12 = dot(vec_1, vec_2)/(norm(vec_1)*norm(vec_2))
cos_sim13 = dot(vec_1, vec_3)/(norm(vec_1)*norm(vec_3))
cos_sim23 = dot(vec_2, vec_3)/(norm(vec_2)*norm(vec_3))
print(f"{cos_sim12=} {cos_sim13=} {cos_sim23=}")

SCALE_PARAM = 2

n_vec_1 = np.array(vec_1, dtype='float32')
n_vec_2 = np.array(vec_2, dtype='float32')
n_vec_3 = np.array(vec_3, dtype='float32')
dist12 = np.linalg.norm(n_vec_1-n_vec_2)
sim12 = math.exp(-((dist12**2)/(2*(SCALE_PARAM**2))))
dist13 = np.linalg.norm(n_vec_1-n_vec_3)
sim13 = math.exp(-((dist13**2)/(2*(SCALE_PARAM**2))))
dist23 = np.linalg.norm(n_vec_2-n_vec_3)
sim23 = math.exp(-((dist23**2)/(2*(SCALE_PARAM**2))))
print(f"{sim12=} {sim13=} {sim23=}")


cv2.imshow('image_1', image_1)
cv2.imshow('image_2', image_2)
cv2.imshow('image_3', image_3)
cv2.waitKey()
