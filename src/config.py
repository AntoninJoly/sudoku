# network params
epoch = 25
patience_early = 3
factor_reduce_lr = 0.1
patience_lr = 2
batch_size = 32
img_size_grid = (512, 512)
img_size_digits = (28, 28)
num_classes = 1

accepted = ['.jpeg', '.jpg', '.png']
dir_data = '../data/json'
grid_model_path = '../models/grid_detection.tflite'
digit_model_path = '../models/digit_classifier.tflite'