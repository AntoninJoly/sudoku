from utils.loss import *
from utils.metrics import *
from generator import *
from model.seg_hrnet import seg_hrnet
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import warnings
warnings.filterwarnings("ignore")

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

# network params
BatchSize = 8
NumChannels = 3
ImgHeight = 512
ImgWidth = 512
NumClass = 1

# training params
GPUs = '0, 1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUs
# Optimizer = 'Adam'  # SGD(lr=0.01, momentum=0.9, nesterov=True)

# data params
trainImageDir = './data/book/train/image/'
valImageDir = './data/book/val/image/'

# visualization params
metric_list = ['acc', 'iou']

model = seg_hrnet(BatchSize, ImgHeight, ImgWidth, NumChannels, NumClass)
# model = multi_gpu_model(model, gpus=2)
model.compile(optimizer='Adam', loss=ce_jaccard_loss, metrics=['accuracy', iou])
# model_path = "seg_hrnet-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}-{val_iou:.4f}.hdf5"
# model_checkpoint = ParallelModelCheckpoint(model, model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=False)
# model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1, save_best_only=False)

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2)
check_point_list = [model_checkpoint, early_stop, reduce_lr]

train_paths, val_paths = get_data_paths(trainImageDir, valImageDir)
train_steps = len(train_paths) // BatchSize
val_steps = len(val_paths) // BatchSize

result = model.fit_generator(generator=batch_generator(train_paths, BatchSize, flag='train'),
                             steps_per_epoch=train_steps,
                             epochs=100,
                             verbose=1,
                             validation_data=batch_generator(val_paths, BatchSize, flag='test'),
                             validation_steps=val_steps,
                             callbacks=check_point_list)

plt.figure()
for metric in metric_list:
    plt.plot(result.epoch, result.history[metric], label=metric)
    plt.scatter(result.epoch, result.history[metric], marker='*')
    val_metric = 'val_' + metric
    plt.plot(result.epoch, result.history[val_metric], label=val_metric)
    plt.scatter(result.epoch, result.history[val_metric], marker='*')
plt.legend(loc='under right')
plt.show()

plt.figure()
plt.plot(result.epoch, result.history['loss'], label="loss")
plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
plt.scatter(result.epoch, result.history['loss'], marker='*')
plt.scatter(result.epoch, result.history['val_loss'], marker='*')
plt.legend(loc='upper right')
plt.show()