import torch
from torchvision.transforms import transforms as T
import argparse
from unet.unet_model import UNet
from torch import optim
from utils.dataset import DatasetVerse
from torch.utils.data import DataLoader

dir_img = 'F:\\Verse_Data\\train_data\\img'
dir_mask = 'F:\\Verse_Data\\train_data\\mask'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标准化到[-1,1],规定均值和标准差
x_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# mask转化为tensor
y_transform = T.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        # minBatch
        step = 0
        for x, y in dataload:
            optimizer.zero_grad()
            inputs = x.to(device)
            labels = y.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 梯度下降，计算梯度
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), "weights_%d.pth" % epoch)
    return model


def train():
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    verse_data = DatasetVerse(dir_img, dir_mask, transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(verse_data, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloader)


def test():
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    verse_data = DatasetVerse(dir_img, dir_mask, transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(verse_data)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()

    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
