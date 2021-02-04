from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 96, 11, 4), nn.ReLU(),
                                  nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(),
                                  nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(), nn.Conv2d(384, 256, 3, 1, 1),
                                  nn.ReLU(), nn.MaxPool2d(3, 2))
        self.fc = nn.Sequential(nn.Linear(25 * 256, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                                nn.Dropout(0.5), nn.Linear(4096, 10))

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

    def predict(self, img):
        return nn.functional.softmax(self.forward(img), dim=1).detach()

    def get_feature(self, img):
        return self.conv(img)
