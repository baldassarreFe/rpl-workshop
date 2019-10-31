import torch

from torch import nn
from torchvision.models import resnet18


class BirdNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        features = self.resnet(input)
        output = self.classifier(features)
        return output

    def train(self, mode=True):
        self.resnet.eval()
        self.classifier.train(mode)

    def eval(self):
        self.resnet.eval()
        self.classifier.eval()
