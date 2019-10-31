from workshop.dataset import CUBDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor
import argparse
from pathlib import Path
from torch.optim import Adam
import torch.nn.functional as F
import tqdm

from workshop.model import BirdNet


def train(args):
    dataset = CUBDataset(
        root_directory=args.datapath,
        set_="train",
        transforms=Compose([
            RandomCrop((224, 224), pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor()
        ])
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers)

    model = BirdNet(num_classes=dataset.number_classes).to(args.device)

    optimizer = Adam(
        params=model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    epoch_bar = tqdm.trange(args.number_epochs, desc='Epoch')
    for epoch in epoch_bar:
        batch_bar = tqdm.tqdm(data_loader, desc='Batch')
        for batch in batch_bar:
            input_batch = batch[0].to(args.device)
            target = batch[1].to(args.device)
            predictions = model(input_batch)
            loss = F.cross_entropy(predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix({'loss': loss.item()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--number_epochs", type=int, required=True)
    parser.add_argument("--number_workers", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args)
