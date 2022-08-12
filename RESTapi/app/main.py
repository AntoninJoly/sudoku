from fastapi import FastAPI, File, UploadFile
import uvicorn
import warnings
warnings.filterwarnings('ignore')
import os
import time
import sys
import base64
import json
import codecs
from utils import *
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from typing import Union, Dict, List
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from starlette.responses import StreamingResponse

logging.basicConfig(filename="event.log",
                    filemode='a',
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level = logging.DEBUG)
logger = logging.getLogger()

class errorMsg(Exception):
    def __init__(self, m):
        self.message = m
    def __str__(self):
        return self.message

class JSONClass(BaseModel):
    size: Union[int,int,int]
    path: str
    img: str

app = FastAPI()

try:
    model_dir = '../models'
    digit_model = load_model(os.path.join(model_dir, 'digit_classifier.tflite'))
    grid_model = load_model(os.path.join(model_dir, 'grid_detection.tflite'))
except Exception as e:
    m = 'Cannot load models'
    logger.error(f'{m}, msg={e}\n########')
    raise(errorMsg(m))

def main(img):

    img_size_grid = (512,512)
    img_size_digit = (28,28)

    try:
        img_in = pad_resize_img(img, img_size_grid)
    except Exception as e:
        m = 'Cannot process input image'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        pred = inference_grid(grid_model, img_in.copy()[:,:,[2,1,0]])
        mask = select_main_grid(pred)

        green = np.zeros(img_in.shape)
        green[mask.astype(bool)] = [0,255,0]
        bitwise = cv2.bitwise_or(img_in.astype(np.uint8), green.astype(np.uint8))
        titles = ['Input image','Prediction', 'Grid ROI']
        vis_segm = visualize_results(img_in, pred, bitwise, titles)
    except Exception as e:
        m = 'Cannot predict mask'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))
    
    try:
        (vis_corner, warp, warp_box), box_warp, h = find_corners_harris(img_in, mask)
        titles = ['Contouring and corners','Homography', 'Grid ROI']
        vis_homography = visualize_results(vis_corner, warp, warp_box, titles)
    except Exception as e:
        m = 'Cannot perform homography'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        (grid_vis, digit_vis, centroid_vis), bbox = analyse_img(warp_box)
        titles = ['Processed grid','Processed digits', 'Grid centroids']
        vis_grid = visualize_results(grid_vis, digit_vis, centroid_vis, titles)
    except Exception as e:
        m = 'Cannot process grid image'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        img_tile, grid_solve = grid_digits_processing(digit_model, digit_vis, bbox)
    except Exception as e:
        m = 'Cannot process digits in grid'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        vis_pred, zeros = solve_sudoku(img_tile, grid_solve)
    except Exception as e:
        m = 'Cannot solve the sudoku'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        res, mask = revert_to_original(grid_solve, zeros, bbox, warp_box, box_warp, h)
        img_in[mask.astype(bool)] = 0
        vis_res = cv2.bitwise_or(img_in.astype(np.uint8), res.astype(np.uint8))[:,:,[2,1,0]]
    except Exception as e:
        m = 'Cannot solve the sudoku'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    return vis_segm, vis_homography, vis_grid, vis_pred, vis_res

def load_image(data, ext):
    if ext in ['.png','.jpeg','.jpg']:
        img = np.array(Image.open(BytesIO(data)))[:,:,[0,1,2]]
    elif ext == '.json':
        json_dict = json.loads(data.decode('utf-8'))
        img = np.fromstring(base64.b64decode(json_dict['img']), np.uint8)
        img = img.reshape(*json_dict['size'])
    return img.astype(np.uint8)

@app.get("/")
async def root():
    return {"Uvicorn": "I'm alive"}

@app.post("/process/")
async def process_image(data: UploadFile):
    _ , ext = os.path.splitext(data.filename)
    img = load_image(await data.read(), ext)
    segm, homo, grid, pred, res = main(img)
    vis  = concatenate_visualization(segm, homo, grid, pred, res)
    _, img_png = cv2.imencode(".png", vis)
    return StreamingResponse(BytesIO(img_png.tobytes()), media_type="image/png")

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
#     json_dict = json.load(open('../test.json'))
#     segm, homo, grid, pred, sudoku_array = main(json_dict)
#     vis = concatenate_visualization(segm, homo, grid, pred)