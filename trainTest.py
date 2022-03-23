import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
loader2 = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
loader3=Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)





def show_batch(loaderList):
    for loader in loaderList:
        for epoch in range(3):
            for step, (batch_x, batch_y) in enumerate(loader):
                # training

                print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    loaderList = {}
    loaderList["t"]=loader
    loaderList["t1"]=loader2
    loaderList["t3"]=loader3
    loaderList.append(loader)
    loaderList.append(loader2)
    loaderList.append(loader3)
    show_batch(loaderList)