import argparse
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
import time

from draw_model import DRAWModel

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='checkpoint/model_final', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=1, help='Number of generated outputs')
parser.add_argument('-t', default=None, help='Number of glimpses.')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Set the number of glimpses.
# Best to just use the same value which was used for training.
params['T'] = int(args.t) if(args.t) else params['T']

# Load the model
model = DRAWModel(params).to(device)
# Load the trained parameters.
model.load_state_dict(state_dict['model'])
print('\n')
print(model)

start_time = time.time()
print('*'*25)
print("Generating Image...")
# Generate images.
with torch.no_grad():
    x = model.generate(int(args.num_output))

with torch.no_grad():
    y = model.generate(int(args.num_output))

x_row = torch.randint(0, 31, (1,1))
x_col = torch.randint(0, 31, (1,1))
y_row = torch.randint(0, 31, (1,1))
y_col = torch.randint(0, 31, (1,1))

# Combining the generated images
z=[]
for i in range(len(x)):
    tx=torch.zeros(3,60,60)
    ty=torch.zeros(3,60,60)
    tx[:, x_row:x_row+28, x_col:x_col+28] = x[i]
    ty[:, y_row:y_row + 28, y_col:y_col + 28] = y[i]
    tz = tx + ty
    z.append(tz)

img_file_name = "Generated_twoDigit_"+datetime.datetime.now().strftime("%d%m%y%H%M%S")

time_elapsed = time.time() - start_time
print('\nDONE!')
print('Time taken to generate image: %.2fs' % (time_elapsed))

print('\nSaving generated image...')
fig = plt.figure(figsize=(int(np.sqrt(int(args.num_output)))*2, int(np.sqrt(int(args.num_output)))*2))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    z[-1], nrow=int(np.sqrt(int(args.num_output))), padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig(img_file_name)
plt.close('all')

img_file_name = img_file_name + ".gif"

# Create animation for the generation.
fig = plt.figure(figsize=(int(np.sqrt(int(args.num_output)))*2, int(np.sqrt(int(args.num_output)))*2))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in z]
anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=2000, blit=True)
anim.save(img_file_name, dpi=100, writer='imagemagick')
print('DONE!')
print('-'*50)
plt.show()