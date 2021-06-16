import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils
import torch.nn as nn

from torchvision import datasets, transforms
from draw_model import DRAWModel
from dataloader import get_data


# Load the checkpoint file.
state_dict = torch.load('checkpoint/model_final')

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']


# Load the model
model = DRAWModel(params).to(device)
# Load the trained parameters.
model.load_state_dict(state_dict['model'])
print('\n')
print(model)

# """
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('test/data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=1)

params['channel'] = 1


# List to hold the losses for each iteration.
# Used for plotting loss curve.
losses = []
latent_lossess = []
iters = 0
avg_loss = 0
avg_latent_loss = 0

print("-" * 25)
print("Starting Testing Loop...\n")
print('Length of Data Loader: %d' % (len(test_loader)))
print("-" * 25)

start_time = time.time()

for epoch in range(1):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(test_loader, 0):
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        
        # Calculate the loss.
        model.forward(data)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(model.cs[-1])
        # Reconstruction loss.
        Lx = criterion(x_recon, data) * model.A * model.B * model.channel
        
        loss_val = Lx.data

        # Latent loss.
        Lz = 0

        for t in range(model.T):
            mu_2 = model.mus[t] * model.mus[t]
            sigma_2 = model.sigmas[t] * model.sigmas[t]
            logsigma = model.logsigmas[t]

            kl_loss = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - 0.5 * model.T
            Lz += kl_loss

        Lz = torch.mean(Lz)

        avg_loss += loss_val
        avg_latent_loss += Lz.data

        # Check progress of testing.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f\tLatent_loss: %.4f'
                  % (epoch + 1, 1, i, len(test_loader), avg_loss / 100, avg_latent_loss/100))
        #
            avg_loss = 0

        losses.append(loss_val)
        latent_lossess.append(Lz.data)
        iters += 1

    avg_loss = 0
    avg_latent_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    

testing_time = time.time() - start_time
print("-" * 50)
print('Testing finished!\nTotal Time for Testing: %.2fm' % (testing_time / 60))
print("-" * 50)


losses_np = np.array(losses)
latent_lossess_np = np.array(latent_lossess)
np.savetxt("test/Test_Reconstruction_lossess.csv", losses_np)
np.savetxt("test/Test_Latent_lossess.csv", latent_lossess_np)

# Plot the testing losses.
plt.figure(figsize=(10, 5))
plt.title("Testing Reconstruction Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("test/Test_ReconstructionLoss_Curve")

# Plot the Latent losses.
plt.figure(figsize=(10, 5))
plt.title("Testing Latent Loss")
plt.plot(latent_lossess)
plt.xlabel("iterations")
plt.ylabel("LatentLoss")
plt.savefig("test/Test_LatentLoss_Curve")