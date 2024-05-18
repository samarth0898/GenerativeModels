import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np 
import matplotlib.pyplot as plt

from fvsbn import FVSBN
def main():

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root = './data', train = True, download = True , transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'

    device = torch.device(device)
    print(device)


    model = FVSBN(784)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 50

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            optimizer.zero_grad()
            images = images.view(images.shape[0], -1)
            images = images.to(device)
            output = model(images)
            output.to(device)
            loss = criterion(output, images)
            loss.backward()
            model.zero_out_lower_tri_gradients()
            optimizer.step()
            total_loss += loss.item()

        # Calculate the average loss for the epoch
        epoch_loss = total_loss/ images.shape[0]
        

        # Print the loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


    new_image = model.sample(0.6,784)
    image_2d = new_image.view(28,28)
    img = image_2d.cpu().detach().numpy()
    plt.figure(figsize=(1, 1))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == "__main__":
    main()
    

