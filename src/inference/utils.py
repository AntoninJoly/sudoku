import numpy as np
import cv2
from itertools import permutations, product
from scipy import ndimage
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

def get_res(img):
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
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    label, _ = ndimage.label(thresh, np.ones((3,3)))
    t = (img.shape[0]*img.shape[1]) / 100

    keep = [key for key,value in Counter(label.flatten()).items() if value > t and key!=0]
    for i in keep:
        label = np.where(label!=i, label, -1)
    gray = np.float32(np.where(label==-1, 0, 255))
    label = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) / 255

    # Detect points that form a line
    gray = np.invert(np.uint8(gray))
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=5, maxLineGap=15)

    # Find intersections
    inter = find_intersections(lines)
    cls = DBSCAN(eps=t/100, min_samples=1).fit(np.array(inter))

    centroid = []
    for i in np.unique(cls.labels_):
        idx = np.where(cls.labels_ == i)[0]
        pt = [inter[i] for i in idx]
        centroid.append([np.mean(np.array(pt)[:,0]).astype(int),np.mean(np.array(pt)[:,1]).astype(int)])
    
    # Draw lines on the image
    img_res = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img_res = cv2.line(img_res, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for idx, coord in enumerate(centroid):
        x, y = coord
        img_res = cv2.circle(img_res, (x, y), 2, (0, 0, 255), -1)

    centroid = np.array(sorted(centroid , key=lambda k: [k[1], k[0]])).reshape(10,10,2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bbox = []
    off = 3
    for i,j in product(np.arange(9), repeat=2):
        x0, x1, y0, y1 = [centroid[i,j][0], centroid[i+1,j+1][0], centroid[i,j][1], centroid[i+1,j+1][1]]
        bbox.append([x0+off, x1, y0+off, y1])

    return centroid, (label, img_res), bbox