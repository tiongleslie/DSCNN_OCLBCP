# ------------------------------------------------------------------------
# Implementation of Dual-stream CNN with multi-feature
# Licensed under The MIT License
# Written by L. Tiong
# ------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import scipy.io as sio
from scipy import spatial
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

# Hide all the warning messages from TensorFlow
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class DSCNN_utils(object):
    def __init__(self):
        print('Model Loading...')

    @staticmethod
    def test_DSCNN(path, test_image_batch_r, test_image_batch_d):
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '/DSCNN.meta')
            saver.restore(sess, path + '/DSCNN')

            x1 = graph.get_tensor_by_name("input_img_rgb:0")
            x2 = graph.get_tensor_by_name("input_img_descriptor:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")

            sum_f = graph.get_tensor_by_name("fc3/add:0")
            max_f = graph.get_tensor_by_name("fc4/add:0")

            test_feed = {x1: test_image_batch_r, x2: test_image_batch_d, keep_prob: 1.0}
            feature1, feature2 = sess.run([sum_f, max_f], feed_dict=test_feed)
            sess.close()

        return feature1, feature2

    @staticmethod
    def recognition(left_feature1, left_feature2, right_feature1, right_feature2):
        # Loading reference set feature vectors
        l_base1 = sio.loadmat('model_result/mat/base/left/feature_list1.mat')
        left_base1 = l_base1['feature1']

        l_base2 = sio.loadmat('model_result/mat/base/left/feature_list2.mat')
        left_base2 = l_base2['feature2']

        r_base1 = sio.loadmat('model_result/mat/base/right/feature_list1.mat')
        right_base1 = r_base1['feature1']

        r_base2 = sio.loadmat('model_result/mat/base/right/feature_list2.mat')
        right_base2 = r_base2['feature2']

        # Calculate score matrix
        left_1 = DSCNN_utils.cosine_similar(left_feature1, left_base1)
        left_2 = DSCNN_utils.cosine_similar(left_feature2, left_base2)
        right_1 = DSCNN_utils.cosine_similar(right_feature1, right_base1)
        right_2 = DSCNN_utils.cosine_similar(right_feature2, right_base2)

        final_score = (left_1 + left_2) + (right_1 + right_2)

        # Loading reference set info
        start = 0
        gallery = sio.loadmat('model_result/mat/gallery.mat')
        gallery = gallery['gallery']
        [x, _] = gallery.shape

        store = np.zeros((x, 2))

        for i in range(x):
            if i == 0:
                endP = int(gallery[i]) - 1
                store[i, 0] = start
                store[i, 1] = endP
            else:
                start += int(gallery[i - 1])
                endP = start + int(gallery[i])
                store[i, 0] = start
                store[i, 1] = endP - 1

        [num, _] = final_score.shape
        final_decision = np.zeros((num, x))
        for i in range(x):
            for j in range(num):
                best_score = final_score[j, int(store[i, 0]):int(store[i, 1]) + 1]
                sorted_score = -np.sort(-best_score, axis=0)
                final_decision[j, i] = np.sum(sorted_score[0:3])

        return final_decision

    @staticmethod
    def cosine_similar(feat_f, base_f):
        [x, _] = feat_f.shape
        scores = np.zeros((x, len(base_f)), dtype=np.float32)

        for i in range(x):
            for j in range(len(base_f)):
                scores[i][j] = 1 - spatial.distance.cosine(feat_f[i], base_f[j])

        return scores

    @staticmethod
    def ref_list():
        with open('model_result/mat/ref.txt') as f:
            names = [line.rstrip() for line in f]

        return names

    @staticmethod
    def display_images(num, result, left_path1, left_path2, right_path1, right_path2, mode=1):
        mpl.rcParams['toolbar'] = 'None'
        f = plt.figure(num+1, figsize=(2.9, 2.55))
        f.patch.set_facecolor('black')
        plt.subplots_adjust(0.15, 0, .85, .8, 0.05, 0.05)

        names = DSCNN_utils.ref_list()
        string = "Sample " + str(num + 1) + "\nRecognised: " + str(names[int(np.argmax(result))]) + "\n\n"
        print(string)

        if mode is 1:
            for ax in f.axes:
                ax.axis('off')
                ax.margins(0, 0)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())

            f.add_subplot(2, 2, 1)
            plt.text(0, 15, string, fontsize=10, color='white')
            img_r_left = Image.open(left_path1)
            plt.imshow(img_r_left)
            plt.gca().set_axis_off()
            img_r_left.close()

            f.add_subplot(2, 2, 3)
            img_d_left = Image.open(left_path2)
            plt.imshow(img_d_left)
            plt.gca().set_axis_off()
            img_d_left.close()

            f.add_subplot(2, 2, 2)
            img_r_right = Image.open(right_path1)
            plt.imshow(img_r_right)
            plt.gca().set_axis_off()
            img_r_right.close()

            f.add_subplot(2, 2, 4)
            img_d_right = Image.open(right_path2)
            plt.imshow(img_d_right)
            plt.gca().set_axis_off()
            img_d_right.close()

            plt.gca().set_axis_off()
            plt.show(block=True)
