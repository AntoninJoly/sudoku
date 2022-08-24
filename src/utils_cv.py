import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

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