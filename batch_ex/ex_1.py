
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


# Hyper Parameter
batch_size = 64
data_height = 28
data_width = 28
channel_n = 1
num_classes = 10

# 방법.1 - Empty Array를 만들고 채워가는 방법
batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
batch_label = np.zeros((batch_size, num_classes))

# 간단한 batch data 만들기
for n, path in enumerate(data_list[:batch_size]):
    image = read_image(path)
    onehot_label = onehot_encode_label(path)
    batch_image[n, :, :, :] = image
    batch_label[n, :] = onehot_label


if __name__ == "__main__":
    print(batch_image.shape, batch_label.shape)

    test_n = 0
    plt.figure()
    plt.title(batch_label[test_n])
    plt.imshow(batch_image[test_n, :, :, 0])
    plt.show()
    print("end")