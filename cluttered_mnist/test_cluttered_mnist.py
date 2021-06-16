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


# Load the checkpoint file.
state_dict = torch.load('checkpoint/model_final')

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

model = DRAW_RAM_Model(params).to(device)
# Load the trained parameters.
model.load_state_dict(state_dict['model'])
print('\n')
print(model)

complete_data = np.load('data/mnist_digit_sample_8dsistortions9x9.npz')
x_test = complete_data['X_test']
y_test = complete_data['y_test']
# """
test_loader = torch.utils.data.DataLoader(
    TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)),
    batch_size=1, shuffle=True)

params['channel'] = 1
# """

# Initialize the model.
model = DRAW_RAM_Model(params).to(device)

criterion = nn.CrossEntropyLoss()

# List to hold the losses for each iteration.
# Used for plotting loss curve.
losses = []
iters = 0
avg_loss = 0

print("-" * 25)
print("Starting Testing Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (
1, 1, len(test_loader)))
print("-" * 25)

start_time = time.time()

for epoch in range(1):
    epoch_start_time = time.time()

    correct = 0
    total=0
    for i, batch_data in enumerate(test_loader, 0):
        data, labels = batch_data
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        
        # Calculate the loss.

        labels = labels.to(device)
        labels = torch.tensor(labels, dtype=torch.long)
        predicted = model(data)

        sof = torch.softmax(predicted, dim=1)
        pred_label = torch.max(sof)

        if pred_label == labels:
            correct += 1

        total += 1

        loss = criterion(predicted, torch.max(labels,1)[0])


        loss_val = loss.data
        avg_loss += loss_val
        

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f\tAccuracy: %.4f'
                  % (epoch + 1, 1, i, len(test_loader), avg_loss / 100, correct/total))

            avg_loss = 0

        losses.append(loss_val)
        iters += 1
    print("Accuracy: {}".format(correct/total))

    avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))


testing_time = time.time() - start_time
print("-" * 50)
print('Testing finished!\nTotal Time for Testing: %.2fm' % (testing_time / 60))
print("-" * 50)
