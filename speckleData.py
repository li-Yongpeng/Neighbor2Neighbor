from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import os
import torch
from torchvision import transforms



def median(img,k_size=3):
    h,w,c=img.shape

    pad=k_size//2
    out=np.zeros((h+2*pad,w+2*pad,c),dtype=float)
    out[pad:pad+h,pad:pad+w]=img.copy().astype(float)

    tmp=out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci]=np.mean(tmp[y:y+k_size,x:x+k_size,ci])

    out=out[pad:pad+h,pad:pad+w]
    return out







class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)

        h=im.size[0]
        w=im.size[1]

        if h<self.patch or w<self.patch:
            im=im.resize((self.patch,self.patch),Image.ANTIALIAS)

        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]




        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]



        # np.ndarray to torch.tensor
        transformer = transforms.Compose([
            #transforms.Resize((self.patch,self.patch)),
            transforms.ToTensor()
        ])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


class speckleDataset(Dataset):
    def __init__(self,noiseSource,noiseTarget,patch=256):
        super(speckleDataset, self).__init__()
        self.source_dir = noiseSource
        self.target_dir=noiseTarget
        self.patch = patch

        self.train_fns = glob.glob(os.path.join(self.source_dir, "*"))
        self.train_fns.sort()
        self.target_fns=glob.glob(os.path.join(self.target_dir,"*"))
        self.target_fns.sort()

        if(len(self.train_fns)!=len(self.target_fns)):
            raise Exception("Training eror: The number of train and target files are not equal")

        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        Tn=self.target_fns[index]
        im = Image.open(fn)
        tm=Image.open(Tn)

        if(im.size !=tm.size):
            raise Exception("The size of train and target images are not equal!")

        im,tm=self.getImage(im,tm)

        # np.ndarray to torch.tensor
        transformer = transforms.Compose([
            # transforms.Resize((self.patch,self.patch)),
            transforms.ToTensor()
        ])
        im = transformer(im)
        tm=transformer(tm)
        return im,tm

    def __len__(self):
        return len(self.train_fns)

    def getImage(self,im,tm):

        h = im.size[0]
        w = im.size[1]

        if h < self.patch or w < self.patch:
            im = im.resize((self.patch, self.patch), Image.ANTIALIAS)
            tm = tm.resize((self.patch, self.patch), Image.ANTIALIAS)


        im = np.array(im, dtype=np.float32)
        tm=np.array(tm,dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]

        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
            tm = tm[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
            tm = tm[:, yy:yy + self.patch, :]


        return im,tm

class valDataset(Dataset):
    def __init__(self,noiseSource,noiseTarget,name):
        super(valDataset, self).__init__()
        self.source_dir = noiseSource
        self.target_dir=noiseTarget
        self.name=name

        self.train_fns = glob.glob(os.path.join(self.source_dir, "*"))
        self.train_fns.sort()
        self.target_fns=glob.glob(os.path.join(self.target_dir,"*"))
        self.target_fns.sort()

        if(len(self.train_fns)!=len(self.target_fns)):
            raise Exception("Testing error: The number of train and target files are not equal")

        print('fetch {} samples for dataset {} testing'.format(len(self.train_fns),self.name))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        Tn=self.target_fns[index]
        im = Image.open(fn)
        tm=Image.open(Tn)

        if (im.size != tm.size):
            raise Exception("The size of train and target images are not equal!")






        im = np.array(im, dtype=np.float32)
        tm = np.array(tm, dtype=np.float32)

        H = im.shape[0]
        W = im.shape[1]
        val_size = (max(H, W) + 31) // 32 * 32

        im = np.pad(
            im,
            [[0, val_size - H], [0, val_size - W], [0, 0]],
            'reflect')
        tm = np.pad(
            tm,
            [[0, val_size - H], [0, val_size - W], [0, 0]],
            'reflect')


        # np.ndarray to torch.tensor
        transformer = transforms.Compose([
            # transforms.Resize((self.patch,self.patch)),
            transforms.ToTensor()
        ])
        im = transformer(im)
        tm=transformer(tm)
        return im,tm

    def __len__(self):
        return len(self.train_fns)



def logTrans(im):

    """
    图像经过对数变换之后，灰度值会从0-255变到0-5.55
    要在8bit的显示器进行显示，需要将其规范到(0,255)之间
    这里在log变换之后给图像乘以一个系数46
    """
    # im = Image.open(srcFile)

    im = np.array(im, dtype=np.float32)
    out = 46 * np.log(1 + im.astype(int)) + 0.5

    # out = Image.fromarray(np.uint8(out))
    # out.save(destFile)

    return out

def expTrans(im):

    im=np.exp(im)-1
    out=Image.fromarray(np.uint8(im))
    out.save("./speckleFiles/log.png")

if __name__=="__main__":
    srcFile="./speckleFiles/test.png"
    destFile="./speckleFiles/out2.png"
    im=Image.open(srcFile)

    im = np.array(im, dtype=np.float32)

    out = np.log(1 + im.astype(int))
    expTrans(out)

    # out=46*np.log(1+im.astype(int))+0.5
    # out=Image.fromarray(np.uint8(out))
    # out.save(destFile)

    print("test")