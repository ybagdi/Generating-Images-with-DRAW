import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils

# from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import TensorDataset

from cluttered_mnist import DRAW_RAM_Model
from dataloader import get_data


# Dictionary storing network parameters.
params = {
    'T': 8,  # Number of glimpses.
    'batch_size': 128,  # 128 Batch size.
    'A': 100,  # 32 Image width
    'B': 100,  # 32 Image height
    # 'z_size': 100,  # Dimension of latent space.
    'read_N': 12,  # N x N dimension of reading glimpse.
    # 'write_N': 5,  # N x N dimension of writing glimpse.
    # 'dec_size': 256,  # Hidden dimension for decoder.
    'enc_size': 256,  # Hidden dimension for encoder.
    'epoch_num': 15,  # Number of epochs to train for.
    'learning_rate': 1e-3,  # Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch': 5,  # After how many epochs to save checkpoints and generate test output.
    'channel': None}  # Number of channels for image.(3 for RGB, etc.)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

params['device'] = device

# train_loader = get_data(params)
params['channel'] = 3

complete_data = np.load('data/mnist_digit_sample_8dsistortions9x9.npz')
x_train = complete_data['X_train']
y_train = complete_data['y_train']
# """
train_loader = torch.utils.data.DataLoader(
    TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)),
    batch_size=params['batch_size'], shuffle=True)

params['channel'] = 1
# """


# Initialize the model.
model = DRAW_RAM_Model(params).to(device)

criterion = nn.CrossEntropyLoss()
# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

# List to hold the losses for each iteration.
# Used for plotting loss curve.
losses = []
iters = 0
avg_loss = 0

print("-" * 25)
print("Starting Training Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (
params['epoch_num'], params['batch_size'], len(train_loader)))
print("-" * 25)

start_time = time.time()

for epoch in range(params['epoch_num']):
    epoch_start_time = time.time()

    for i, batch_data in enumerate(train_loader, 0):
        data, labels = batch_data
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        optimizer.zero_grad()
        # Calculate the loss.

        labels = labels.to(device)
        labels = torch.tensor(labels, dtype=torch.long)
        predicted = model(data)
        loss = criterion(predicted, torch.max(labels,1)[0])
        # loss_val = loss.cpu().data.numpy()
        loss_val = loss.data
        avg_loss += loss_val
        # Calculate the gradients.
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters.
        optimizer.step()

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch + 1, params['epoch_num'], i, len(train_loader), avg_loss / 100))

            avg_loss = 0

        losses.append(loss_val)
        iters += 1

    avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    # Save checkpoint and generate test output.
    if (epoch + 1) % params['save_epoch'] == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params
        }, 'checkpoint/model_epoch_{}'.format(epoch + 1))

        # with torch.no_grad():
        #     generate_image(epoch + 1)

training_time = time.time() - start_time
print("-" * 50)
print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
print("-" * 50)
# Save the final trained network paramaters.
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'params': params
}, 'checkpoint/model_final'.format(epoch))
