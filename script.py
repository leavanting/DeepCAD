import os
import time
import sys

flag = sys.argv[1]

###################################################################################################################################################################
# Only train
if flag == 'train':
    # for train
    os.system('python train.py --datasets_folder TrainingData --n_epochs 50 --GPU 1 --lr 0.00005 --img_h 100 --img_w 100 --img_s 300 --train_datasets_size 1200')

# Test all data
if flag == 'test-all':
    # for test
    os.system('python test.py --denoise_model GoodModel-2 --datasets_folder FinalData-midNA --GPU 1 \
                              --img_h 146 --img_w 146 --img_s 200 --gap_h 100 --gap_w 100 --gap_s 120\
                              --test_datasize 20000')
###################################################################################################################################################################
# Only test 1000 slices for model screening
if flag == 'test-small':
    # for test
    os.system('python test.py --denoise_model model --datasets_folder TestData --GPU 1 \
                              --img_h 90 --img_w 90 --img_s 200 --gap_h 60 --gap_w 60 --gap_s 120\
                              --test_datasize 1000')
###################################################################################################################################################################
# Train and then test only 1000 slices for model screening
if flag == 'train-and-test':

    os.system('python train.py --datasets_folder TrainingData --n_epochs 40 --GPU 1 --lr 0.0001 --img_h 100 --img_w 100 --img_s 300 --train_datasets_size 1500')

    os.system('python test.py --denoise_model TrainingData_20210324_2147 --datasets_folder FinalData --GPU 1 \
                              --img_h 146 --img_w 146 --img_s 200 --gap_h 100 --gap_w 100 --gap_s 120\
                              --test_datasize 1000')
