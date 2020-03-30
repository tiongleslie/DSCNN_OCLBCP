# ------------------------------------------------------------------------
# Implementation of Image Util
# Licensed under The MIT License
# Written by L. Tiong
# ------------------------------------------------------------------------
import natsort
import numpy as np
import os
from PIL import Image


class image_utils(object):
    def __init__(self, path, input_type, mode='left', w=80, h=80, c=3, class_label=None, trained=True):
        self.path = path
        self.width = w
        self.height = h
        self.channel = c
        self.class_label = class_label
        self.mode = mode
        self.trained = trained
        self.input_type = input_type

    def load_data(self, shuffle_list=None):
        path_list = os.listdir(self.path + "/" + self.input_type)
        path_list.sort()
        path_array = []
        s_path_array = []
        label_list = []
        label_v = 0
        labels = None

        if self.mode is None:
            print("Please assign 'left' or 'right' as mode")
            exit(1)
        elif self.mode is 'left' or self.mode is 'right':
            for class_name in path_list:
                directFiles = os.listdir(self.path + "/" + self.input_type + "/" + class_name + "/" + self.mode + "/")
                files = natsort.natsorted(directFiles)
                for file in files:
                    jpg = self.path + "/" + self.input_type + "/" + class_name + "/" + self.mode + "/" + file
                    path_array.append(jpg)
                    label_list.append(label_v)
                label_v += 1

        if shuffle_list is None:
            print('Reading~')

            shuffle_list = list(range(0, len(path_array)))
            if self.trained is True:
                np.random.shuffle(shuffle_list)

            if self.class_label is not None:
                labels = np.zeros([len(path_array), self.class_label])

            for i in range(len(shuffle_list)):
                s_path_array.append(path_array[shuffle_list[i]])
                if self.class_label is not None:
                    labels[i, label_list[shuffle_list[i]]] = 1
        else:
            print('Reading~')

            for i in range(len(shuffle_list)):
                s_path_array.append(path_array[shuffle_list[i]])

        if self.class_label is None:
            return s_path_array, shuffle_list
        else:
            return s_path_array, labels, shuffle_list

    def read_images_by_batch(self, batch_arr, batch_size):
        data_list = []

        for i in range(len(batch_arr)):
            img = Image.open(batch_arr[i])
            arr = np.array(img).reshape((self.height, self.width, self.channel))
            data_list.append(arr)
            img.close()

        X = np.array(data_list)
        data_X = np.zeros((batch_size, self.height, self.width, self.channel), dtype=np.float32)

        for i in range(batch_size):
            data_X[i, :, :, :] = X[i, :, :, :]

        return data_X / 255.
