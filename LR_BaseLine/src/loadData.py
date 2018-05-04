import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import struct
from sklearn.preprocessing import normalize

# 训练集文件
train_images_idx3_ubyte_file = 'E:/DataSet/mnist/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'E:/DataSet/mnist/train-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images.T


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def getLoadBinaryClassData():
    path = "E:\\WorkStation\\LR\\"
    dataX = pd.read_csv(path+"trainX.csv")
    dataY = pd.read_csv(path+"trainY.csv", header=None)
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(dataX).T
    Y = np.concatenate((np.mat(dataY).T, 1-np.mat(dataY).T), axis=0)
#     Y = mat([mat(dataY).T, 1 - mat(dataY).T])
    return (X, Y)

def getMnist49(path):
    # def testRun():
    train_images = load_train_images(path+"train-images.idx3-ubyte").T #(num_rows*num_cols,num_images)
    b = np.ones(train_images.shape[0])
    train_images = np.column_stack((train_images,b))
    train_labels = load_train_labels(path+"train-labels.idx1-ubyte")
    print("train_images shape", train_images.shape)
    print("train+labels shape", train_labels.shape)
    selectIndex = np.logical_or(train_labels==4, train_labels==9)
    train_49_images = train_images[selectIndex,:]
    train_49_labels = train_labels[selectIndex]
    print(train_49_images.shape)
    print(train_49_labels.shape)
    X = np.mat(train_49_images).T
    X = np.mat(normalize(X, axis=1, norm='l2'))
    Y = np.mat(np.mat(train_49_labels==4).astype(int))
    return (X, Y)