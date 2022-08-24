import unittest
import sys
import warnings
sys.path.append('../src')
from utils import *
import config as cfg
import numpy as np

class Test_SegmentationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None: # class method to ensure that the initialization is done only once
        warnings.simplefilter("ignore", ResourceWarning)
        cls.data, cls.mask = process_data(cfg.dir_data, cfg.img_size_grid)
        cls.path = [os.path.join(cfg.dir_data, i) for i in os.listdir(cfg.dir_data)]

    def test_img_dimensions(self):
        img_shape = Test_SegmentationDataset.data.shape
        self.assertEqual(img_shape[1:], (*cfg.img_size_grid, 3))

    def test_mask_dimensions(self):
        mask_shape = Test_SegmentationDataset.mask.shape
        self.assertEqual(mask_shape[1:], (*cfg.img_size_grid, 1))

    def test_segmentation_mask_values(self):
        unique_values_mask = np.unique(Test_SegmentationDataset.mask)
        self.assertEqual(unique_values_mask.tolist(), [0,1])

    def test_img_data_matching(self):
        idx = np.random.randint(0, len(Test_SegmentationDataset.path))
        img_json, mask_json = process_json(Test_SegmentationDataset.path[idx], cfg.img_size_grid)
        img, mask = Test_SegmentationDataset.data[idx], Test_SegmentationDataset.mask[idx]
        self.assertTrue(np.array_equal(img, img_json))
        self.assertTrue(np.array_equal(mask, mask_json))

    def test_predict_grid(self):
        idx = np.random.randint(0, len(Test_SegmentationDataset.path))
        img = Test_SegmentationDataset.data[idx]
        model = load_model(cfg.grid_model_path)
        mask = inference_grid(model, img)

        self.assertEqual(mask.shape, cfg.img_size_grid)

    # def test_full_cycle(self):

    # def test_iou_segmentation(self):

    # def test_accuracy_classifier(self):

    # def test_


if __name__ == '__main__':
    unittest.main()