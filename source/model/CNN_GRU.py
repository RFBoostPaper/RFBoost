import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import AlexNet
class CNN_GRU(nn.Module):
    # 1d-CNN + GRU
    def __init__(self, input_channel, input_size, num_label, n_gru_hidden_units=128, f_dropout_ratio=0.5, batch_first=True):
        super(CNN_GRU, self).__init__()
        # [@, T, C, F]

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channel, 16, kernel_size=5, stride=1, padding="same"),
            nn.LayerNorm([16, input_size]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.LayerNorm([32, input_size // 2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.LayerNorm([64, input_size // 4]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            
            nn.Linear(64 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(f_dropout_ratio),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
        )
        # [@, T, 128]
        self.rnn = nn.GRU(input_size=128, hidden_size=n_gru_hidden_units, batch_first=batch_first)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=f_dropout_ratio),
            nn.Linear(n_gru_hidden_units, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=f_dropout_ratio),
            nn.Linear(64, num_label),
        )

    def forward(self, input):
        # [@, T, C]
        cnn_out_list = [self.cnn(input[:, t, :, :]) for t in range(input.size(1))]
        cnn_out = torch.stack(cnn_out_list, dim=1)
        # [@, T, 128]
        out, _ = self.rnn(cnn_out)
        x = out[:, -1, :]
        x = self.classifier(x)
        return x

def main():
    # [@, T, 1, F]
    input = torch.zeros((16, 256, 1, 128)).cuda()
    model = CNN_GRU(input_channel = 1, num_label = 6, n_gru_hidden_units=128, f_dropout_ratio=0.5).cuda()
    o = model(input)
    summary(model, input_size=input.size()[1:])
    print(o.size())

if __name__ == '__main__':
	main()