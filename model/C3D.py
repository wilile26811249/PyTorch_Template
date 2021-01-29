import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    """
    Paper Title:
        Learning Spatiotemporal Features with 3D Convolutional Network

    Paper Link:
        https://arxiv.org/pdf/1412.0767.pdf

    Input data shape:
        (batch_size, channels, frames, height, width)
    """
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

    def init_weights(self):
        """
        Initiate the parameters either from existing checkpoint or from
        scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 1)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.pool1(output)

        output = F.relu(self.conv2(output))
        output = self.pool2(output)

        output = F.relu(self.conv3a(output))
        output = F.relu(self.conv3b(output))
        output = self.pool3(output)

        output = F.relu(self.conv4a(output))
        output = F.relu(self.conv4b(output))
        output = self.pool4(output)

        output = F.relu(self.conv5a(output))
        output = F.relu(self.conv5b(output))
        output = self.pool5(output)

        output = output.view(-1, 8192)
        output = F.relu(self.fc6(output))
        output = nn.Dropout(p = 0.5)(output)

        output = self.fc7(output)
        output = F.relu(output)
        output = nn.Dropout(p = 0.5)(output)

        logics = self.fc8(output)
        probs = nn.Softmax()(logics)

        return probs