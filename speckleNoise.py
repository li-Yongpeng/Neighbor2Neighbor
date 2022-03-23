import numpy as np
import torch
from PIL import Image
import cv2
import os

def median(image):
    mean_image = np.zeros(shape=image.shape, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                mean_image[i][j][k] = np.mean(image[i:i + 2, j:j + 2, k])
    return mean_image

def median2(img,k_size=3):
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


# 使用PIL
def addSpeckle(img,var=1):
    w, h = img.size
    c = len(img.getbands())


    # z = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(n, 2)).view(np.complex128)
    z = np.random.multivariate_normal(np.zeros(2), 0.5 * var * np.eye(2), size=(h, w, c)).view(np.complex128)
    #torch.repeat
    #torch.repeat
    noise = z
    noise_abs = abs(noise)
    noise_abs=np.squeeze(noise_abs)
    noise_abs=median2(noise_abs)


    noiseimage=np.multiply(np.array(img),noise_abs)
    noise_img = np.clip(noiseimage, 0, 255).astype(np.uint8)
    return Image.fromarray(noise_img)

# 给图像添加斑点噪声
def speckNoise():
    srcPath = "/home/lyp/PycharmProjects/Neighbor2Neighbor/validation/BSD300/test"
    destPath = "/home/lyp/Disk/N2N/BSD/1"

    if not os.path.exists(destPath):
        os.makedirs(destPath)

    i = 0
    for root, dirs, files in os.walk(srcPath):
        for file in files:
            srcName = os.path.join(root, file)
            img = Image.open(srcName)
            imgout = addSpeckle(img, var=1)
            destName = os.path.join(destPath, file)
            imgout.save(destName)
            print(i)
            i += 1
def getTrainTset():
    srcPath = "/home/lyp/dataset/ssdd/SSDD数据以及标签/JPEGImages"

    trainList = []
    testList = []
    for root, dirs, files in os.walk(srcPath):
        for file in files:
            file = file.split('.')[0]
            if file.endswith("1") or file.endswith("9"):
                testList.append(file)
            else:
                trainList.append(file)
    print("训练集大小: " + str(len(trainList)))
    print("测试集大小: " + str(len(testList)))
    trainFile = "/home/lyp/dataset/ssdd/SSDD数据以及标签/train.txt"
    testFile = "/home/lyp/dataset/ssdd/SSDD数据以及标签/test.txt"

    with open(trainFile, "a+") as f:
        for l in trainList:
            f.write(l)
            f.write('\r\n')

    with open(testFile, "a+") as f:
        for l in testList:
            f.write(l)
            f.write('\r\n')

    print("test")

if __name__ == "__main__":
    src="/home/lyp/dataset/ssdd/JPEGImages/000150.jpg"
    img=Image.open(src)
    sd=np.array(img)
    c=len(img.getbands())
    print(c)
