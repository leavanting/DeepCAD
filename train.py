import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import scipy.io as scio
from network import Network_3D_Unet
from tensorboardX import SummaryWriter
import numpy as np
from data_process import shuffle_datasets, train_preprocess_lessMemoryMulStacks, shuffle_datasets_lessMemory
from utils import save_yaml
from skimage import io

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--GPU', type=int, default=0, help="the index of GPU you will use for computation")

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_s', type=int, default=300, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=64, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=64, help="the height of image sequence")

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_folder', type=str, default='DataForPytorch', help="A folder containing files for training")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--select_img_num', type=int, default=10000, help='select the number of images')
parser.add_argument('--train_datasets_size', type=int, default=1500, help='datasets size for training')
opt = parser.parse_args()

# default image gap is 0.75*image_dim
opt.gap_s=int(opt.img_s*0.75)
opt.gap_w=int(opt.img_w*0.75)
opt.gap_h=int(opt.img_h*0.75)

print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)

########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = opt.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_path = opt.output_dir + '/' + current_time
pth_path = 'pth//'+ current_time
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

yaml_name = pth_path+'//para.yaml'
save_yaml(opt, yaml_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
batch_size = opt.batch_size
lr = opt.lr

name_list, noise_img, coordinate_list = train_preprocess_lessMemoryMulStacks(opt)
# print('name_list -----> ',name_list)
########################################################################################################################
L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()
########################################################################################################################
denoise_generator = Network_3D_Unet(in_channels = 1,
                                out_channels = 1,
                                final_sigmoid = True)
if torch.cuda.is_available():
    print('\033[1;31mUsing GPU for training...\033[0m')
    denoise_generator.cuda()
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()

cuda = True if torch.cuda.is_available() else False

########################################################################################################################
optimizer_G = torch.optim.Adam( denoise_generator.parameters(), 
                                lr=opt.lr, betas=(opt.b1, opt.b2))

########################################################################################################################
prev_time = time.time()
time_start=time.time()
# start training
for epoch in range(0, opt.n_epochs):
    name_list = shuffle_datasets_lessMemory(name_list)
    # print('name list -----> ',name_list) 
    for index in range(len(name_list)):
        single_coordinate = coordinate_list[name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch1 = noise_img[init_s:end_s:2,init_h:end_h,init_w:end_w]
        noise_patch2 = noise_img[init_s+1:end_s:2,init_h:end_h,init_w:end_w]
        real_A = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch1, 3),0)).cuda()
        real_A = real_A.permute([0,4,1,2,3])
        real_B = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch2, 3),0)).cuda()
        real_B = real_B.permute([0,4,1,2,3])
        # print('real_A shape -----> ',real_A.shape)
        # print('real_B shape -----> ',real_B.shape)
        input_name = name_list[index]
        real_A = Variable(real_A)
        fake_B = denoise_generator(real_A)
        # Pixel-wise loss
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)

        ################################################################################################################
        optimizer_G.zero_grad()
        # Total loss
        Total_loss =  0.5*L1_loss + 0.5*L2_loss
        Total_loss.backward()
        optimizer_G.step()

        ################################################################################################################
        batches_done = epoch * len(name_list) + index
        batches_left = opt.n_epochs * len(name_list) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        ################################################################################################################
        if index%1 == 0:
            time_end=time.time()
            print('\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] ETA: %s, Time cost: %.2d s        ' 
            % (
                epoch+1,
                opt.n_epochs,
                index+1,
                len(name_list),
                Total_loss.item(),
                L1_loss.item(),
                L2_loss.item(),
                time_left,
                time_end-time_start
            ), end=' ')

        if (index+1)%len(name_list) == 0:
            print('\n', end=' ')

        ################################################################################################################
        #save model 
        if (index+1)%(len(name_list)) == 0:
            torch.save(denoise_generator.state_dict(), pth_path + '//G_' + str(epoch) +'_iter_'+ str(index) + '.pth')