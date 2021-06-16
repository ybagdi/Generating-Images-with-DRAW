import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import torchvision.datasets as dsets

def generate_test_images():
    data = np.load('data/mnist_digit_sample_8dsistortions9x9.npz')
    x_test = data['X_test']
    # x_test = x_test.reshape(1000, 40, 40)
    y_test = data['y_test']

    for i in range(x_test.shape[0]):
        index = y_test[i][0]
        im_file_path = "data/test_data_for_user_1/" + str(index) + "/" + str(i)

        plt.figure(figsize=(1, 1))
        plt.axis("off")
        plt.imshow(x_test[i])
        plt.savefig(im_file_path)
        plt.close()

#only for visualizing the test images
# generate_test_images()

class DRAW_RAM_Model(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']
        self.A = params['A']
        self.B = params['B']
        self.read_N = params['read_N']
        self.enc_size = params['enc_size']
        self.device = params['device']
        self.channel = params['channel']

        # Stores the read image for each time step.
        self.cs = [0] * self.T

        # self.encoder = nn.LSTMCell(2*self.read_N*self.read_N*self.channel + self.enc_size, self.enc_size)
        # self.encoder = nn.LSTMCell(2 * self.read_N * self.read_N * self.channel, self.enc_size)
        self.encoder = nn.LSTMCell(self.read_N * self.read_N * self.channel, self.enc_size)

        self.fc_layer_0 = nn.Linear(self.enc_size, self.enc_size)
        self.fc_layer = nn.Linear(self.enc_size, 10)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.enc_size, 5)

    def forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)

        for t in range(self.T):
            # c_prev = torch.zeros(self.batch_size, self.B * self.A * self.channel, requires_grad=True,
            #                      device=self.device) if t == 0 else self.cs[t - 1]
            # Equation 3.
            # x_hat = x - torch.sigmoid(c_prev)
            x_hat = x
            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(x, x_hat, h_enc_prev)
            # Equation 5.
            # h_enc, enc_state = self.encoder(torch.cat((r_t, h_enc_prev), dim=1), (h_enc_prev, enc_state))
            h_enc, enc_state = self.encoder(r_t, (h_enc_prev, enc_state))
            # Equation 8.
            # self.cs[t] = c_prev #+ r_t  


            h_enc_prev = h_enc
            # h_dec_prev = h_dec

        # category = torch.softmax(self.fc_layer(h_enc_prev), dim=1)
        temp = self.fc_layer_0(h_enc_prev)
        relu = nn.ReLU()
        temp = relu(temp)
        category = self.fc_layer(temp)
        # category = self.fc_layer(self.fc_layer_0(h_enc_prev))
        # category = torch.log_softmax(self.fc_layer(h_enc_prev),dim=1)
        return category


    def read(self, x, x_hat, h_dec_prev):
        # Using attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(self.channel, 2)
            if self.channel == 3:
                img = img.view(-1, 3, self.B, self.A)
            elif self.channel == 1:
                img = img.view(-1, self.B, self.A)

            # Equation 27.
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1, self.read_N*self.read_N*self.channel)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        # return torch.cat((x, x_hat), dim=1)
        return x



    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22.
        gx = (self.A + 1) / 2 * (gx_ + 1)
        # Equation 23
        gy = (self.B + 1) / 2 * (gy_ + 1)
        # Equation 24.
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta_)
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(start=0.0, end=N, device=self.device, requires_grad=True, ).view(1, -1)

        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device, requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device, requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)

            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    # def loss(self, x, label):
    #     predicted_label = self.forward(x)
    #     # for i in range(predicted_label.shape[0]):
    #
    #     # predicted_label = torch.argmax(predicted_label)
    #     # l = nn.CrossEntropyLoss()
    #     l = nn.NLLLoss()
    #     return l(predicted_label, label)