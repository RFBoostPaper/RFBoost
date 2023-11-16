import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import AlexNet

class Widar3(nn.Module):
    # CNN + GRU
    def __init__(self, input_shape, input_channel, num_label, n_gru_hidden_units=128, f_dropout_ratio=0.5, batch_first=True):
        super(Widar3, self).__init__()
        self.num_label = num_label
        self.n_gru_hidden_units = n_gru_hidden_units
        self.f_dropout_ratio = f_dropout_ratio
        # [@, T_MAX, 1, 20, 20] 
        self.input_shape = input_shape
        self.input_time = input_shape[1]
        self.input_channel = input_shape[2]
        self.input_x, self.input_y = input_shape[3], input_shape[4]

        self.Tconv1_out_channel=16
        self.Tconv1_kernel_size=5
        self.Tconv1_stride=1

        self.Tdense1_out = 64
        self.Tdense2_out = 64
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, self.Tconv1_out_channel, self.Tconv1_kernel_size, self.Tconv1_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(self.Tconv1_out_channel * ((self.input_x - 4) // 2) * ((self.input_y - 4) // 2), self.Tdense1_out),
            nn.ReLU(inplace=True),
            nn.Dropout(f_dropout_ratio),
            nn.Linear(self.Tdense1_out, self.Tdense2_out),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=self.Tdense2_out, hidden_size=n_gru_hidden_units, batch_first=batch_first)
        self.dropout2 = nn.Dropout(f_dropout_ratio)
        # [Pytorch] DO NOT USE SOFTMAX HERE!!!
        self.dense3 = nn.Linear(n_gru_hidden_units, num_label)
        
    def forward(self, input):
        # [@, T_MAX, 1, 20, 20]
        cnn_out_list = [self.cnn(input[:, t, :, :, :]) for t in range(self.input_time)]
        cnn_out = torch.stack(cnn_out_list, dim=1)
        # [@, T_MAX, 64]
        out, _ = self.gru(cnn_out)
        x = out[:, -1, :]
        x = self.dropout2(x)
        x = self.dense3(x)
        # x = F.relu(x)
        return x

def main():
    input = torch.zeros((4, 38, 1, 20, 20)).cuda()
    model = Widar3(input.shape, input_channel = 1, num_label = 6, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()
