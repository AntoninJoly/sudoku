import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
from itertools import permutations, product
from scipy import ndimage
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from math import atan2
import tensorflow as tf

def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter
    
def inference_grid(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.expand_dims(img,axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return np.squeeze(np.rint(output_data))

def inference_digit(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.expand_dims(img.reshape(28,28,1), axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(output_data, axis=-1)[0]
    return  idx + 1, output_data[0][idx]

def select_main_grid(p):
    label, _ = ndimage.label(p, np.ones((3,3)))
    m = Counter(label.flatten()).most_common(2)[1][0]
    p = np.where(label==m, 1, 0)
    return p

def perp(a):
    b = np.empty_like(a)
    b[0], b[1] = -a[1], a[0]
    return b

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def segment(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def intersect(a1,a2,b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    det = np.dot(perp(da), db)
    if det != 0 and segment(a1,a2,b1,b2): # lines intersects
        num = np.dot(perp(da), dp)
        res = (num / det.astype(float))*db + b1
        return list(res.astype(int))
    else:
        return [0,0]

def scale(pts, factor=0.1):
    x1, y1, x2, y2 = pts
    t0, t1 = 0.5*(1.0 - factor), 0.5*(1.0 + factor)
    
    x1 = int(x1 + (x1 - x2) * t0)
    y1 = int(y1 + (y1 - y2) * t0)
    x2 = int(x2 + (x2 - x1) * t1)
    y2 = int(y2 + (y2 - y1) * t1)

    return x1, y1, x2, y2

def sort_points(pts):
    pts_sort = pts[np.argsort(pts[:,1])]
    pts_sort = np.array([pts_sort[np.argmin(pts_sort[:2,0])],
                         pts_sort[np.argmax(pts_sort[:2,0])],
                         pts_sort[np.argmax(pts_sort[2:,0])+2],
                         pts_sort[np.argmin(pts_sort[2:,0])+2]])
    ratio = (pts_sort[2,0] - pts_sort[3,0]) / (pts_sort[1,0] - pts_sort[0,0])
    return pts_sort, ratio

def warp_grid(img, pts):
    
    pts_src, ratio = sort_points(pts)
    
    pts_dst = np.array([[100,100],[400,100],[400,400],[100,400]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    height, width = img.shape[:2]
    dst = cv2.warpPerspective(img.astype(np.uint8), h, (width,height))
    
    offset = 10
    x0,x1 = np.min(pts_dst[:,0]) - offset, np.max(pts_dst[:,0]) + offset
    y0,y1 = np.min(pts_dst[:,1]) - offset, np.max(pts_dst[:,1]) + offset

    return (x0,x1,y0,y1), dst, h

def find_corners_harris(img_in, mask):
    cnt, hierarchy = cv2.findContours(np.expand_dims(mask.copy(), axis=-1).astype(np.uint8),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
    
    gray = cv2.dilate(mask.astype(np.uint8), (5,5), iterations = 1).astype(np.float32)
    dst = cv2.cornerHarris(gray, 25, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
    
    approx = cv2.approxPolyDP(centroids.astype(int)[1:], 0.1*cv2.arcLength(cnt[0],True), True)
    centroids = sorted(list(np.squeeze(approx)), key=lambda x:[x[0],x[1]])
    
    corner = cv2.drawContours(img_in.copy(), cnt, -1, (0, 255, 0), 3)
    for i in centroids:
        corner = cv2.circle(corner, (i[0], i[1]), 3, (255,0,0), 3)
    
    (x0,x1,y0,y1), warp, h = warp_grid(img_in.copy(), np.array(centroids))
    return (corner, warp, warp[y0:y1,x0:x1]), ([x0, y0], [x1,y1]), h

def pad_resize_img(img, img_size):
    h,w,_ = img.shape
        
    if h>w:
        offset = h-w
        left, right = np.floor(offset/2).astype(int), np.ceil(offset/2).astype(int)
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, (0,0,0))
    elif h<w:
        offset = w-h
        top, bot = np.floor(offset/2).astype(int), np.ceil(offset/2).astype(int)
        img = cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
        
    img = cv2.resize(img, img_size)
    return img.astype(int)

def pad_resize_gray(img, img_size):
    h,w = img.shape
        
    if h>w:
        offset = h-w
        left, right = np.floor(offset/2).astype(int), np.ceil(offset/2).astype(int)
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, 0)
    elif h<w:
        offset = w-h
        top, bot = np.floor(offset/2).astype(int), np.ceil(offset/2).astype(int)
        img = cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, 0)
        
    img = cv2.resize(img, img_size)
    return img.astype(int)

def find_intersections(lines):
    inter = []
    for i in permutations(lines, 2): # Test all combinations of lines
        x1_1, y1_1, x1_2, y1_2 = scale(i[0][0])
        x2_1, y2_1, x2_2, y2_2 = scale(i[1][0])
        
        p1 = np.array([x1_1, y1_1])
        p2 = np.array([x1_2, y1_2])
        p3 = np.array([x2_1, y2_1])
        p4 = np.array([x2_2, y2_2])

        inter.append(intersect(p1,p2,p3,p4))
    return [i for i in inter if i!=[0,0]]

def sort_centroid_x(c, thresh):
    c = sorted(c , key=lambda k: k[1])
    x = [-1]+list(np.where(np.diff(np.array(c)[:,1])>thresh)[0])+[len(c)]
    row = [sorted(c[x[idx]+1:x[idx+1]+1], key=lambda k: k[0]) for idx in range(len(x)-1)]
    return row

def sort_centroid_y(c, thresh):
    c = sorted(c , key=lambda k: k[0])
    y = [-1]+list(np.where(np.diff(np.array(c)[:,0])>thresh)[0])+[len(c)]
    col = [sorted(c[y[idx]+1:y[idx+1]+1], key=lambda k: k[1]) for idx in range(len(y)-1)]
    return col

def bbox_from_centroid(centroid):
    bbox, c = [], np.array(centroid.copy()).reshape(10,10,2)
    for i,j in product(np.arange(9), repeat=2):
        x0, x1, y0, y1 = [c[i,j][0], c[i+1,j+1][0], c[i,j][1], c[i+1,j+1][1]]
        bbox.append([x0+3, x1, y0+3, y1])
    return bbox

def find_centroid_from_lines(inter, t, h, w):
    cls = DBSCAN(eps=t/20, min_samples=1).fit(np.array(inter))
    centroid = []
    for i in np.unique(cls.labels_):
        pt = [inter[i] for i in np.where(cls.labels_ == i)[0]]
        x,y = np.mean(np.array(pt)[:,0]).astype(int),np.mean(np.array(pt)[:,1]).astype(int)
        if all([x>=0,y>=0,x<=w, y<=h]):
            centroid.append([x,y])
    return centroid

def check_centroid_grid(pts):

    row = sort_centroid_x(pts, 5)
    mean_x, mean_y = [], []
    centroid = []

    for r in row:
        if len(r)>1:
            segdists = np.sqrt((np.diff(r, axis=0) ** 2).sum(axis=1))
            # angle = np.array([atan2(r[i][1] - r[i-1][1],r[i][0] - r[i-1][0]) * 180 / np.pi for i in range(1,len(r))])
            mean_x.append(np.mean(segdists))
            centroid.append(r)
    col = sort_centroid_y([i for s in centroid for i in s], 5)
    
    centroid = []
    for c in col:
        if len(c)>1:
            segdists = np.sqrt((np.diff(r, axis=0) ** 2).sum(axis=1))
            mean_y.append(np.mean(segdists))
            centroid.append(c)
    
    centroid = sort_centroid_x([i for s in centroid for i in s], 5)

    return [i for s in centroid for i in s], []
    # return [i for s in centroid for i in s], np.array([i for s in r for i in s]).astype(int)

def label_island(thresh, num):
    label, _ = ndimage.label(thresh, np.ones((3,3)))
    t = (thresh.shape[0]*thresh.shape[1]) / num
    keep = [key for key,value in Counter(label.flatten()).items() if value > t and key!=0]
    for i in keep:
        label = np.where(label!=i, label, -1)
    gray = np.where(label==-1, 0, 255).astype(np.uint8)
    return gray, t

def split_digits_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    # Remove digits in the grid
    gray, t = label_island(thresh, 500)
    grid = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) / 255

    # Remove grid and keep digits
    digit = thresh.astype(bool).astype(np.uint8)
    digit[np.invert(gray).astype(bool)] = 0
    digit, _ = label_island(digit, 1250)
    digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2RGB)
    
    return grid, np.invert(gray), digit, t
    
def analyse_img(img):
    # Detect the grid in the image
    grid, gray_grid, digit, t = split_digits_grid(img)
    # Detect points that form a line
    lines = cv2.HoughLinesP(gray_grid,
                            rho=1.0,
                            theta=np.pi/180,
                            threshold=150,
                            minLineLength=5,
                            maxLineGap=50)
    
    # Filter lines by angular offset
    angle = np.array([abs(atan2(y2-y1,x2-x1) * 180 / np.pi) for (x1,y1,x2,y2) in np.squeeze(lines)])
    lines = lines[[i<10 or (i > 80 and i < 100) for i in angle]]

    # Find intersections & centroids & bboxes
    inter = find_intersections(lines)
    pts = find_centroid_from_lines(inter, t, *img.shape[:2])
    centroid, r = check_centroid_grid(pts)

    if len(centroid)==100:
        bbox = bbox_from_centroid(centroid)
    else:
        bbox = []
    
    # Draw results on the image
    img_res = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img_res = cv2.line(img_res, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for idx, coord in enumerate(centroid):
        img_res = cv2.circle(img_res, coord, 2, (0, 0, 255), -1)
        img_res = cv2.putText(img_res, f'{idx}',coord,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1,cv2.LINE_4)
        
    return (grid, digit, img_res), bbox

def concatenate_visualization(segm, homo, grid, pred, res):
    res = cv2.resize(res, (1500,1500))
    left, right = int(np.floor((2000-1500)/2)), int(np.ceil((2000-1500)/2))
    res = cv2.copyMakeBorder(res,0,0,left, right,cv2.BORDER_CONSTANT, None, value = (255,255,255))
    return np.concatenate((segm, homo, grid, pred, res), axis=0)

def visualize_results(img1, img2, img3, titles):
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.title(titles[0])
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.title(titles[1])
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.title(titles[2])
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    vis = data.reshape((int(h), int(w), -1))[:, :, [2, 1, 0]]
    ax = np.where(vis!=[255,255,255])[0]
    plt.close()
    return vis[ax[0]:ax[-1],:]

def grid_digits_processing(digit_model, digit_vis, bbox, sudoku_array = []):
    
    grid_solve = np.empty((81)).astype(int)
    img_tile = np.zeros((81, 28, 28))

    for idx, box in enumerate(bbox):
        offset = 0
        x0, x1, y0, y1 = box
        roi = digit_vis[y0+offset:y1-offset,x0+offset:x1-offset]
        gray = cv2.cvtColor(np.invert(roi), cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, np.zeros(gray.shape), 0, 255, cv2.NORM_MINMAX)
        img_resize = pad_resize_gray(gray, (28,28))
        
        pred, confidence = inference_digit(digit_model, img_resize)
        l = pred if np.sum(gray)>100 else 0
        grid_solve[idx] = l
        img_tile[idx,:,:] = img_resize

    return img_tile, grid_solve.reshape(9,9)

def num_on_img(l):
    img_solution = np.zeros((28,28,3))
    if l!=0:
        img_solution = cv2.putText(img_solution, f'{l}', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img_solution.astype(np.uint8), cv2.COLOR_RGB2GRAY)

def solve_sudoku(img_tile, grid_solve):
    def isSafe(grid_solve, row, col, num):
        for x in range(9):
            if grid_solve[row][x] == num:
                return False
        for x in range(9):
            if grid_solve[x][col] == num:
                return False
        startRow = row - row % 3
        startCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid_solve[i + startRow][j + startCol] == num:
                    return False
        return True

    def solve(grid_solve, row, col):
        if (row == 9 - 1 and col == 9):
            return True
        if col == 9:
            row += 1
            col = 0
        if grid_solve[row][col] > 0:
            return solve(grid_solve, row, col + 1)
        for num in range(1, 10):
            if isSafe(grid_solve, row, col, num):
                grid_solve[row][col] = num
                if solve(grid_solve, row, col + 1):
                    return True
            grid_solve[row][col] = 0
        return False

    zeros = np.where(grid_solve==0, True, False)

    if not (solve(grid_solve, 0, 0)):
        print("Solution does not exist")        
    
    fig = plt.figure(figsize=(20,20))

    for idx, (p,s) in enumerate(zip(grid_solve.reshape(-1), zeros.reshape(-1))):

        plt.subplot(9,9,idx+1)
        if s:
            img = num_on_img(p)
        else:
            img = img_tile[idx]
            t = f'Predicted - {p}'
            plt.title(t, fontsize=15)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    vis = data.reshape((int(h), int(w), -1))[:, :, [2, 1, 0]]
    plt.close()

    return vis, zeros

def revert_to_original(grid_solve, zeros, bbox, img, box_warp, h):
    val = grid_solve[zeros].reshape(-1)
    box = np.array(bbox)[zeros.reshape(-1)]

    for v, b in zip(val, box):
        x0, x1, y0, y1 = b
        x,y = x0+6,y0+22
        label = v if v != 0 else '?'
        img = cv2.putText(img, f'{label}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_4)
    ([left,top],[x1,y1]) = box_warp
    bot, right = 512 - y1, 512 - x1
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, None, value = (0,0,0))
    
    h_inv = np.linalg.inv(h)
    img = cv2.warpPerspective(img, h_inv, (512,512))

    return img