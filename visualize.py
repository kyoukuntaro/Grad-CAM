import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda')
    main_dir = 'Directori Path'
    file_name = 'File name'
    img = Image.open(main_dir + file_name)
    img = img.resize((224,224), Image.BICUBIC)
    x = np.zeros([1,3,224,224])
    x[0,:,:,:] = (np.asarray(img, dtype=np.float)/255).transpose([2,0,1])
    x = (torch.Tensor(x)).to(device, dtype=torch.float)

    model = torch.load('best_model.pth')
    model.eval()
    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h_ = model.maxpool(h)
    grad0 = Variable(h_, requires_grad=True)
    h = model.layer1(grad0)
    h = model.layer2(h)
    h = model.layer3(h)
    h0 = model.layer4(h)
    h1 = model.avgpool(h0)[:,:,0,0]
    grad = Variable(h1, requires_grad=True)
    y = model.fc(grad)
    cls_predicted = torch.argmax(y,1)
    y = y[:,cls_predicted[0]]
    y.backward()
    #print(grad0.grad)
    result = torch.Tensor(np.zeros([1,1,7,7])).to(device, dtype=torch.float)

    relu = nn.ReLU()
    for i in range(45):
        result += relu(h0[:,[i],:,:] * grad.grad[cls_predicted,i])
    result = result.cpu().detach().numpy()
    result = result[0,0,:,:].astype('uint')
    image = Image.fromarray(result)
    image = image.resize((224,224),Image.BICUBIC)
    image = np.asarray(image)
    plt.title('result: class {0}'.format(cls_predicted.item()))
    plt.imshow(image,cmap='bwr')
    plt.show()


if __name__ == '__main__':
    main()