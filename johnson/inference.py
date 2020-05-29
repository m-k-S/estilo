import torch
import torchvision.transforms as transforms

from PIL import Image

from network import FastNeuralStyle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = 'weights.pth'
target_path = 'honolulu.jpg'
img_size = 256

model = FastNeuralStyle()
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)

transform = transforms.Compose([
    # transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

with torch.no_grad():
    img = transform(Image.open(target_path)).unsqueeze(0)
    y = model(img).squeeze()
    img = y.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save("output.jpg")
