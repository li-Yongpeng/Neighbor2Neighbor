from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet
from vgg16 import Vgg16
from vgg16 import FeatureLoss

# 拆分数据文件和训练文件
from speckleData import speckleDataset,valDataset
from speckleData import validation_kodak,validation_Set14,validation_bsd300


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./SpeckleData/VOC2007/1')
parser.add_argument('--target_dir',type=str,default="./Imagenet_val")
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--val_target_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./resultsSP')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=220)
parser.add_argument('--n_snapshot', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=12)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--Lambda3", type=float, default=0.1 )
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument("--slist1",nargs='+',type=int,default=[1,1,1,1])
parser.add_argument("--slist2",nargs='+',type=int,default=[1,1,1,1])


opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


# Training Set
TrainingDataset = speckleDataset(opt.data_dir, opt.target_dir,patch=opt.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)


# Validation Set
# Kodak_dir = os.path.join(opt.val_dirs, "Kodak")
Kodak_dir="/home/lyp/Disk/N2N/Kodak/1"
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
Set14_dir = os.path.join(opt.val_dirs, "Set14")

# Kodak_dir1 = os.path.join(opt.val_target_dirs, "Kodak")
Kodak_dir1="/home/lyp/PycharmProjects/Neighbor2Neighbor/validation/Kodak"
BSD300_dir1 = os.path.join(opt.val_target_dirs, "BSD300")
Set14_dir1= os.path.join(opt.val_target_dirs, "Set14")

KodakDataset=valDataset(Kodak_dir,Kodak_dir1,"Kodak")
BSD300Dataset=valDataset(BSD300_dir,BSD300_dir1,"BSD300")
Set14Dataset=valDataset(Set14_dir,Set14_dir1,"Set14")

KodakLoader=DataLoader(dataset=KodakDataset,
                            num_workers=4,
                            batch_size=2,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
BSD300Loader=DataLoader(dataset=BSD300Dataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
Set14Loader=DataLoader(dataset=Set14Dataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
dataLoaderList=[]
dataLoaderList.append(KodakLoader)
# dataLoaderList.append(BSD300Loader)
# dataLoaderList.append(Set14Loader)




# Network
network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
featureNet=Vgg16()

if opt.parallel:
    network = torch.nn.DataParallel(network)
    featureNet=torch.nn.DataParallel(featureNet)

network = network.cuda()
featureNet=featureNet.cuda()

fLoss=FeatureLoss(opt.slist1, opt.slist2)

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

checkpoint(network, 0, "model")
print('init finish')

for epoch in range(1, opt.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, (noisy, clean) in enumerate(TrainingLoader):
        st = time.time()


        clean=torch.log(clean+1)
        noisy=torch.log(noisy+1)

        clean=clean/5.55
        noisy=noisy/5.55

        clean = clean.cuda()
        noisy=noisy.cuda()

        optimizer.zero_grad()

        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        with torch.no_grad():
            noisy_denoised = network(noisy)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = network(noisy_sub1)
        noisy_target = noisy_sub2
        Lambda = epoch / opt.n_epoch * opt.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss1 = torch.mean(diff**2)
        loss2 = Lambda * torch.mean((diff - exp_diff)**2)

        loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2


        # with torch.no_grad():
        #     outImage=featureNet(noisy_output)
        #     inImage=featureNet(noisy,True)
        #     l1Loss=torch.mean(fLoss.featureMSE(outImage,inImage))
        #     l2Loss=torch.mean(fLoss.featureGRAM(outImage,inImage))
        #     loss3=l1Loss+l2Loss
        #
        #
        # loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2 +opt.Lambda3*loss3

        loss_all.backward()
        optimizer.step()

        # print(
        #     '{:04d} {:05d} Loss1={:.4f}, Lambda={}, Loss2={:.4f}, LossMSE={:.4f}, LossGram={:.4f},Loss_Full={:.4f}, Time={:.4f}'
        #     .format(epoch, iteration, np.mean(loss1.item()), Lambda,
        #             np.mean(loss2.item()), np.mean(l1Loss.item()),np.mean(l2Loss.item()),np.mean(loss_all.item()),
        #             time.time() - st))

        print(
            '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss_all.item()),
                        time.time() - st))


    scheduler.step()

    torch.cuda.empty_cache()

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        checkpoint(network, epoch, "model")
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)

        dataListName=["Kodak","BSD300","Set14"]

        for i in range(len(dataLoaderList)):
            psnr_result = []
            ssim_result = []
            for iteration, (noisy, clean) in enumerate(dataLoaderList[i]):

                noisy = torch.log(noisy + 1)
                noisy=noisy/5.55
                noisy = noisy.cuda()

                with torch.no_grad():
                    prediction = network(noisy)
                    prediction=prediction*5.55
                    prediction=torch.exp(prediction)-1

                prediction = prediction.permute(0, 2, 3, 1)
                prediction=prediction.cpu().numpy()
                pred255 = np.clip(prediction, 0,
                                  255).astype(np.uint8)

                clean=clean.permute(0,2,3,1)
                clean=clean.numpy()
                for itr in range(pred255.shape[0]):
                    cur_psnr = calculate_psnr(clean[itr].astype(np.float32),
                                              pred255[itr].astype(np.float32))
                    psnr_result.append(cur_psnr)
                    cur_ssim = calculate_ssim(clean[itr].astype(np.float32),
                                              pred255[itr].astype(np.float32))
                    ssim_result.append(cur_ssim)
                    if epoch == opt.n_snapshot:
                        save_path = os.path.join(
                            validation_path,
                            "{:03d}-{:04d}-{:03d}_denoised.png".format(
                                iteration,itr, epoch))
                        Image.fromarray(pred255[itr]).convert('RGB').save(save_path)

            psnr_result = np.array(psnr_result)
            avg_psnr = np.mean(psnr_result)
            avg_ssim = np.mean(ssim_result)

            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(dataListName[i]))
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))


