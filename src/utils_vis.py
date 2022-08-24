import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
    mask = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    mask[mask!=0]=1
    mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=1)
    img[mask!=1] = 0
    return img, mask