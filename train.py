import torch
from torch import nn
from torchvision.models import resnet34
from torch.optim import Adam;

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np


class Image_Dataset(Dataset):
    def __init__(self, file_ls):
        super(Image_Dataset, self).__init__()
        self.main_dir = 'Directori Name'
        self.files = pd.read_csv(file_ls)
        self.size = len(self.files.index)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(self.files.loc[idx, 'file_name'])
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)
        img = img.transpose([2, 0, 1]) / 255
        return img, self.files.loc[idx, 'target']


def main():
    lr = 0.01
    cls = 45

    train_dataset = Image_Dataset('Train File List CSV')
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataset = Image_Dataset('Test File List CSV')
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    device = torch.device('cuda')

    model = resnet34(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=45)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_v = []

    for epoch in range(20):
        accuracy_v = []
        model.train()
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = (data[0].to(device, dtype=torch.float))
            y = model(x)
            loss = criterion(y, data[1].to(device))
            loss.backward()
            optimizer.step()
            loss_v.append(loss.item())
            if (len(loss_v) == 10):
                print('train loss: ', np.mean(loss_v), 'train accuracy: ', np.mean(accuracy_v))
                loss_v = []
                # print('train accuracy: ', np.mean(accuracy_v))
                accuracy_v = []
            predict = torch.argmax(y, 1)
            accuracy_v.append(float(sum((predict == data[1].to(device))).item() / float(predict.shape[0])))

        accuracy_v = []
        model.eval()
        for i, data in enumerate(test_dataloader):
            x = (data[0].to(device, dtype=torch.float))
            y = model(x)
            predict = torch.argmax(y, 1)
            accuracy_v.append(float(sum((predict == data[1].to(device))).item() / float(predict.shape[0])))
            # print(float(sum((predict == data[1].to(device))).item() /float(predict.shape[0])))
        print('test accuracy: ', np.mean(accuracy_v))
        torch.save(model, 'model{0}.pth'.format(str(epoch).zfill(2)))


if __name__ == '__main__':
    main()