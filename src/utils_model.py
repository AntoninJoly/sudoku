import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import config as cfg

def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def inference_grid(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.expand_dims(img, axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return np.squeeze(np.rint(output_data))

def inference_digit(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.expand_dims(img.reshape(*cfg.img_size_digits,1), axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(output_data, axis=-1)[0]
    return  idx + 1, output_data[0][idx]