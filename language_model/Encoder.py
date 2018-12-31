import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    """
        A Convolutional Neural Network Model
    """

    def __init__(self, num_classes=10):
        """ setup convolutional net """
        super(Encoder, self).__init__()

        ## --------------------------

        #resnet = models.resnet152(pretrained=True)
        #modules = list(resnet.children())[:-1]      # delete the last fc layer.
        #self.resnet = nn.Sequential(*modules)

        # add first layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # add second layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        ## --------------------------

        # self.linear = nn.Linear(resnet.fc.in_features, num_classes)
        self.linear = nn.Linear(18*18*32, num_classes)

        ## --------------------------

        self.bn = nn.BatchNorm1d(num_classes, momentum=0.01)



    def forward(self, images):
        """Extract feature vectors from input images."""

        with torch.no_grad():
            #features = self.resnet(images)
            features = self.layer1(images)
            features = self.layer2(features)

        features = features.reshape(out.size(0), -1)

        # features = self.fc(features)
        features = self.bn(self.linear(features))

        return features





















# ----
