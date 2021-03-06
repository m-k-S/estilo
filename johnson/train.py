from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim

from PIL import Image

import network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = 'train_imgs'
style_img_path = 'starrynight.jpg'
img_size = 256
batch_size = 4
lr = 1e-3
epochs = 1
style_weight = 1e6
content_weight = 1e3

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
def vgg_norm(batch, mean=norm_mean, std=norm_std):
    b, c, h, w = batch.shape
    std = std.repeat(1, b).view(b, 3, 1, 1)
    mean = mean.repeat(1, b).view(b, 3, 1, 1)
    return (batch - mean) / std

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
# train_dataset = torch.utils.data.Subset(train_dataset, [i for i in range(20000)])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

FNS = network.FastNeuralStyle().to(device)
LossNet = network.LossNetwork().to(device)
optimizer = optim.Adam(FNS.parameters(), lr=lr)

style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

style_img = vgg_norm(style_transform(Image.open(style_img_path)).unsqueeze(0).to(device))
style_feats = LossNet(style_img)
style_layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '22': 'relu4_2', '26': 'relu4_4', '35': 'relu5_4'}
content_layer = 'relu2_2'

losses = []
FNS.train()
for epoch in range(epochs):
    print("========Epoch {}/{}========".format(epoch+1, epochs))
    iter = 1
    for batch, _ in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        yhat = FNS(batch)

        batch_content_loss = 0
        batch_style_loss = 0
        for b in range(len(batch)):
            content_feats = LossNet(vgg_norm(batch[b, :, :, :].unsqueeze(0)))
            yhat_feats = LossNet(vgg_norm(yhat[b, :, :, :].unsqueeze(0)))

            # relu2_2 for content loss
            content_loss = network.content_loss(yhat_feats[content_layer], content_feats[content_layer])
            batch_content_loss += content_loss

            style_loss = 0
            for layer in style_layers.values():
                style_loss += network.style_loss(yhat_feats[layer], style_feats[layer])
            batch_style_loss += style_loss

        total_loss = content_weight * batch_content_loss + style_weight * batch_style_loss
        losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

        if iter % 50 == 0:

            print ("ITER {} LOSS: {}".format(iter, losses[-1]))
            torch.save(FNS.state_dict(), 'weights.pth')


        iter += 1
