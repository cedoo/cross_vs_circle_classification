import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torchmetrics
from timeit import default_timer as timer
from tqdm.auto import tqdm

class RuchamPsaJakSra(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class RuchamPsaJakSraKonwulsyjny(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16, 
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    
class RuchamPsaJakSraDrugi(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

device = "cpu"
BATCH_SIZE = 64
train_dir = Path("data/train/")
test_dir = Path("data/test/")
add_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Grayscale(),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder(root=train_dir, transform=add_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=add_transform)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=0,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              num_workers=0,
                              shuffle=True)

train_image_batch, train_label_batch = next(iter(train_dataloader))

input_shape = len(nn.Flatten()(train_image_batch[0])[0])
hidden_units = 32
output_shape = len(train_data.classes)

model_0 = RuchamPsaJakSraKonwulsyjny(1, hidden_units, output_shape)
model_0.to("cpu")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

epochs = 3
train_time_start_on_cpu = timer()
if __name__ == '__main__':
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n-------")
        ### Training
        train_loss = 0
        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_dataloader):
            model_0.train() 
            # 1. Forward pass
            # print(f"trainig tensor: {X}")
            # print(f"shape: {X.shape}")
            y_pred = model_0(X)

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            train_loss += loss # accumulatively add up the loss per epoch 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print out how many samples have been seen
            if (batch+1) % 45 == 0:
                print(f"Looked at {(batch+1) * len(X)}/{len(train_dataloader.dataset)} samples")
        train_loss /= len(train_dataloader)
        print(f"average train loss per epoch: {train_loss:.10f}")

        test_loss, test_acc = 0, 0
        model_0.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_pred = model_0(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += torchmetrics.functional.accuracy(test_pred, y, task="multiclass", num_classes=output_shape)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)/100
        print(f"Test loss: {test_loss:.5f}")
        print(f"Test accuracy: {test_acc:.2f}%")

    train_time_end_on_cpu = timer()
    total_train_time = print_train_time(start=train_time_start_on_cpu, 
                                        end=train_time_end_on_cpu,
                                        device=str(next(model_0.parameters()).device))

    torch.save(obj=model_0.state_dict(),f="model/model.pth")

# Test on specific image

# test_image_batch, test_label_batch = next(iter(test_dataloader))
# single_image, single_label = test_image_batch[0], test_label_batch[0]
# print(single_image)
# with Image.open("data/guess/guess.bmp") as f:
#     single_image = add_transform(f)
#     model_0.eval()
#     with torch.inference_mode():
#         pred = model_0(single_image)

#     print(f"Output logits:\n{pred}\n")
#     percentages = torch.softmax(pred, dim=1)
#     for i, sign in enumerate(train_data.classes):
#         print(f"It's {percentages[0][i]*100:.2f}% {sign}")
#     print(f"Guess: {train_data.classes[torch.argmax(percentages)]}")
# print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
# print(f"Output prediction label:\n{train_data.classes[torch.argmax(torch.softmax(pred, dim=1), dim=1)[0]]}\n")
# print(f"Actual label:\n{train_data.classes[single_label]}")


# print(nn.Flatten()(train_image_batch[0]).shape)
# print(train_image_batch[0])
# plt.imshow(train_image_batch[0].permute(1, 2, 0))
# plt.show(block=True)

# image_path_list = list(train_dir.glob("*/*.bmp"))

# print(image_path.parent.stem)
# img = Image.open(image_path)
# img.show()

# data_transform = transforms.Compose([
#     # Resize the images to 64x64
#     transforms.Resize(size=(8, 8)),
#     # Flip the images randomly on the horizontal
#     transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
#     # Turn the image into a torch.Tensor
#     transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
# ])

# def plot_transformed_images(image_paths, transform, n=3, seed=42):
#     random.seed(seed)
#     random_image_paths = random.sample(image_paths, k=n)
#     for image_path in random_image_paths:
#         with Image.open(image_path) as f:
#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(f) 
#             ax[0].set_title(f"Original \nSize: {f.size}")
#             ax[0].axis("off")

#             transformed_image = transform(f).permute(1, 2, 0) 
#             ax[1].imshow(transformed_image) 
#             ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
#             ax[1].axis("off")

#             fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
#             print(transformed_image)
#             plt.show(block=True)

# plot_transformed_images(image_path_list, 
#                         transform=data_transform, 
#                         n=3)