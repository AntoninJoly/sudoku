from fastapi import FastAPI
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
import io
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

app = FastAPI()

try:
    model_dir = '../models'
    digit_model = load_model(os.path.join(model_dir, 'digit_classifier.tflite'))
    grid_model = load_model(os.path.join(model_dir, 'grid_detection.tflite'))
except Exception as e:
    m = 'Cannot load models'
    logger.error(f'{m}, msg={e}\n########')
    raise(errorMsg(m))

json_dict = json.load(open('../test.json'))

def main(json_dict):

    img_size_grid = (512,512)
    img_size_digit = (28,28)
    logger.info('Start')
    img = np.fromstring(base64.b64decode(json_dict['img']), np.uint8)
    img = img.reshape(*json_dict['size'])
    logger.info('Could read json file')

    try:
        img_in = pad_resize_img(img, img_size_grid)
    except Exception as e:
        m = 'Cannot process input image'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    try:
        pred = inference_grid(grid_model, img_in)
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
        (vis_corner, warp, warp_box), corner_warp = find_corners_harris(img_in, mask)
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
        vis_pred, sudoku_array = grid_digits_processing(digit_model, digit_vis, bbox)
    except Exception as e:
        m = 'Cannot process digits in grid'
        logger.error(f'{m}, msg={e}\n########')
        raise(errorMsg(m))

    return vis_segm, vis_homography, vis_grid, vis_pred, sudoku_array

class JSONClass(BaseModel):
    size: Union[int,int,int]
    path: str
    img: str

@app.get("/")
async def root():
    return {"Uvicorn": "I'm alive"}

@app.post("/process/")
async def process_image():
    segm, homo, grid, pred, sudoku_array = main(json_dict)
    vis = concatenate_visualization(segm, homo, grid, pred)
    res, img_png = cv2.imencode(".png", vis)
    return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
#     json_dict = json.load(open('../test.json'))
#     segm, homo, grid, pred, sudoku_array = main(json_dict)
#     vis = concatenate_visualization(segm, homo, grid, pred)