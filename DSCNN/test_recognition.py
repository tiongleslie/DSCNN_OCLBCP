# ------------------------------------------------------------------------
# Example Code for Periocular in the wild: OC-LBCP and Dual-stream CNN
# Licensed under The MIT License
# Written by L. Tiong
# ------------------------------------------------------------------------
import argparse
from image_utils import image_utils
from DSCNN_utils import DSCNN_utils
import os
import sys
import tensorflow.python.util.deprecation as deprecation

# Hide all the warning messages from TensorFlow
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', help='Test samples dataset path.', default='test_data')
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=64)
    parser.add_argument('--display_images', type=int,
                        help='Demonstrate the results by entering 1 as True and 0 as False.]', default=1)

    return parser.parse_args(argv)


def main(args):
    batch_size = args.batch_size
    test_path = args.test_data

    ##
    ## Test Data
    ##
    print("======================================")
    print("============ Data Loading ============")
    print("======================================")

    ##
    ## Left data
    ##
    # Read the images from R
    left_raw_data = image_utils(path=test_path, input_type='rgb', mode='left', trained=False)
    left_valid_data_r, shuffle_list = left_raw_data.load_data()
    print('RAW images of left ocular (test_data) are loaded')

    # Read the images from R
    left_des_data = image_utils(path=test_path, input_type='descriptor', mode='left', trained=False)
    left_valid_data_d, _ = left_des_data.load_data(shuffle_list=shuffle_list)
    print('Descriptor images left ocular (test_data) are loaded')

    ##
    ## Right data
    ##
    # Read the images from R
    right_raw_data = image_utils(path=test_path, input_type='rgb', mode='right', trained=False)
    right_valid_data_r, shuffle_list = right_raw_data.load_data()
    print('RAW images of right ocular (test_data) are loaded')

    # Read the images from R
    right_des_data = image_utils(path=test_path, input_type='descriptor', mode='right', trained=False)
    right_valid_data_d, _ = right_des_data.load_data(shuffle_list=shuffle_list)
    print('Descriptor images right ocular (test_data) are loaded')

    #
    # Model
    #
    print("\n\n=======================================")
    print("============ Model Loading ============")
    print("=======================================")

    batch_arr_valid = range(0, len(left_valid_data_r), batch_size)
    max_valid = batch_arr_valid[len(batch_arr_valid) - 1]

    for k in range(len(batch_arr_valid)):
        if batch_arr_valid[k] == max_valid:
            left_test_range_r = left_valid_data_r[batch_arr_valid[k]:len(left_valid_data_r)]
            left_test_range_d = left_valid_data_d[batch_arr_valid[k]:len(left_valid_data_d)]
            left_test_image_batch_r = left_raw_data.read_images_by_batch(left_test_range_r, len(left_test_range_r))
            left_test_image_batch_d = left_des_data.read_images_by_batch(left_test_range_d, len(left_test_range_d))

            right_test_range_r = right_valid_data_r[batch_arr_valid[k]:len(right_valid_data_r)]
            right_test_range_d = right_valid_data_d[batch_arr_valid[k]:len(right_valid_data_d)]
            right_test_image_batch_r = right_raw_data.read_images_by_batch(right_test_range_r, len(right_test_range_r))
            right_test_image_batch_d = right_des_data.read_images_by_batch(right_test_range_d, len(right_test_range_d))
        else:
            left_test_range_r = left_valid_data_r[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            left_test_range_d = left_valid_data_d[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            left_test_image_batch_r = left_raw_data.read_images_by_batch(left_test_range_r, batch_size)
            left_test_image_batch_d = left_des_data.read_images_by_batch(left_test_range_d, batch_size)

            right_test_range_r = right_valid_data_r[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            right_test_range_d = right_valid_data_d[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            right_test_image_batch_r = right_raw_data.read_images_by_batch(right_test_range_r, batch_size)
            right_test_image_batch_d = right_des_data.read_images_by_batch(right_test_range_d, batch_size)

        left_feature1, left_feature2 = DSCNN_utils.test_DSCNN(path="model_result/save/left",
                                                              test_image_batch_r=left_test_image_batch_r,
                                                              test_image_batch_d=left_test_image_batch_d)
        right_feature1, right_feature2 = DSCNN_utils.test_DSCNN(path="model_result/save/right",
                                                                test_image_batch_r=right_test_image_batch_r,
                                                                test_image_batch_d=right_test_image_batch_d)
        result = DSCNN_utils.recognition(left_feature1, left_feature2, right_feature1, right_feature2)

        print("Completed test!")

        print("\n\n======================================")
        print("======== Recognition Results =========")
        print("======================================")
        print("Number of samples tested: %i\n" % (len(result)))

        [num, _] = result.shape
        for j in range(num):
            DSCNN_utils.display_images(j, result[j], left_valid_data_r[j], left_valid_data_d[j], right_valid_data_r[j],
                                       right_valid_data_d[j], mode=args.display_images)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
