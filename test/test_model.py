import torch

from workshop.model import BirdNet


def test_birdnet(device):
    batch_size = 4
    num_classes = 7

    input = torch.rand(batch_size, 3, 224, 224).to(device)
    model = BirdNet(num_classes=num_classes).to(device)
    output = model(input)

    assert output.device == device
    assert output.ndimension() == 2
    assert output.shape[0] == batch_size
    assert output.shape[1] == num_classes

    output.mean().backward()
    
    assert model.resnet.conv1.weight.grad is None
    assert model.classifier.weight.grad is not None


def test_train_eval():
    model = BirdNet(num_classes=3)

    model.train()
    assert not model.resnet.training
    assert model.classifier.training

    model.eval()
    assert not model.resnet.training
    assert not model.classifier.training
