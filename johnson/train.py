from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim

from PIL import Image

import network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = 'train_imgs'
style_img_path = 'starrynight.jpg'
img_size = 256
batch_size = 1
lr = 1e-3
epochs = 1
style_weight = 1
content_weight = 100

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
style_img = transform(Image.open(style_img_path)).unsqueeze(0).to(device)

FNS = network.FastNeuralStyle().to(device)
LossNet = network.LossNetwork(style_img).to(device)
optimizer = optim.Adam(FNS.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    print("========Epoch {}/{}========".format(epoch+1, epochs))
    iter = 1
    for batch, _ in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        yhat = FNS(batch)

        _, c_loss, s_loss = LossNet(yhat, batch)
        total_loss = content_weight * c_loss + style_weight * s_loss
        losses.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            print ("CURRENT LOSS: {}".format(losses[-1]))
            output = yhat.clone().squeeze()
            output = transforms.ToPILImage()(output)
            output.save("output.jpg")

        iter += 1
