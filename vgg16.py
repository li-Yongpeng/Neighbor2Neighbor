import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import numpy

# net=models.vgg16(pretrained=True)
# print(net)

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        # 加载vgg16模型
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        features = list(self.model.features)[:23]

        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()
        self.avgPool=nn.AvgPool2d(2)

    def forward(self, x,pool=False):
        if pool:
            x=self.avgPool(x)
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)


# feature Loss
def featureMSE(a,b):
    result=torch.zeros(a[0].shape[0],dtype=a[0].dtype)
    para=[1,1,1,1]
    para = torch.as_tensor(para,dtype=a[0].dtype)
    for i in range(len(a)):
        temp=a[i]-b[i]
        temp=temp.view(temp.size()[0],-1)
        pixNum=temp[0].numel()
        temp=temp.pow(2)
        temp=torch.sum(temp,dim=1)
        temp=temp/pixNum
        temp=temp.cpu()

        result+=para[i]*temp
    result=torch.abs(result)
    return result

def gramMatrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)  # C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)  # bmm 将features与features_t相乘
    return gram



# feature style loss using gram
def featureGRAM(A,B):
    (b,c,h,w)=A[0].size()
    result=torch.zeros(b,dtype=A[0].dtype)

    para=[1,1,1,1]

    for i in range(len(A)):

        featuresA=gramMatrix(A[0])
        featuresB=gramMatrix(B[0])
        temp=featuresA-featuresB

        temp=temp.view(b,-1)
        temp = torch.sum(temp, dim=1)
        temp=temp.cpu()

        result+=para[i]*temp
    result = torch.abs(result)
    return result

def featureLoss(outImage,inImage):
    l1=featureMSE(outImage,inImage)
    l2=featureGRAM(outImage,inImage)
    return l1+l2


class FeatureLoss:
    def __init__(self,argList1,argList2):
        self.featurePara=argList1
        self.gramPara=argList2

    def featureMSE(self,a, b):
        result = torch.zeros(a[0].shape[0], dtype=a[0].dtype)
        para = self.featurePara
        para = torch.as_tensor(para, dtype=a[0].dtype)
        for i in range(len(a)):
            temp = a[i] - b[i]
            temp = temp.view(temp.size()[0], -1)
            pixNum = temp[0].numel()
            temp = temp.pow(2)
            temp = torch.sum(temp, dim=1)
            temp = temp / pixNum
            temp = temp.cpu()

            result += para[i] * temp
        result = torch.abs(result)
        return result

    def featureGRAM(self,A, B):
        (b, c, h, w) = A[0].size()
        result = torch.zeros(b, dtype=A[0].dtype)

        para = self.gramPara

        for i in range(len(A)):
            featuresA = gramMatrix(A[0])
            featuresB = gramMatrix(B[0])
            temp = featuresA - featuresB

            temp = temp.view(b, -1)
            temp = torch.sum(temp, dim=1)
            temp = temp.cpu()

            result += para[i] * temp
        result = torch.abs(result)
        return result



if __name__=="__main__":
    a = torch.rand((2, 3, 128, 128))
    b=torch.rand((2,3,256,256))

    net=Vgg16()
    a1=net(a)
    b1=net(b,True)

    c=featureLoss(a1,b1)

    print(c)