
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

data_list = glob('C:\\git\\AI_S\\batch_ex\\data\\trainingSet\\*\\*.jpg')

label_name_list = []
def get_label_from_path(path):
    return int(path.split('\\')[-2])

for path in data_list:
    label_name_list.append(get_label_from_path(path))
unique_label_names = np.unique(label_name_list)

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label

label_list = [onehot_encode_label(path).tolist() for path in data_list]

def read_image(path):
    image = np.array(Image.open(path))
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(image.shape[0], image.shape[1], 1)
