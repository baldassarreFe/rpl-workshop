import random
import string

from workshop.dataset import CUBDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Resize
import argparse
from pathlib import Path
from torch.optim import Adam
import torch.nn.functional as F
import torch
import tqdm
from time import time
from torch.utils.tensorboard import SummaryWriter

from workshop.model import BirdNet
import pyaml


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def get_average(self):
        if isinstance(self.average, torch.Tensor):
            return float(self.average.cpu().detach())
        return self.average

    def update(self, value, num):
        self.value = value
        self.sum += value * num
        self.count += num
        self.average = self.sum / self.count

    def __repr__(self):
        return f"{self.get_average():.4f}"


def train(args):
    writer = SummaryWriter(log_dir=args.logdir)

    # Datasets
    dataset_tr = CUBDataset(
        root=args.datapath,
        train=True,
        transforms=Compose([
            Resize(256),
            RandomCrop((224, 224), pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor()
        ])
    )
    data_loader_tr = DataLoader(
        dataset_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers)

    dataset_val = CUBDataset(
        root=args.datapath,
        train=False,
        transforms=Compose([
            CenterCrop(224),
            ToTensor()
        ])
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers)

    # Model
    model = BirdNet(num_classes=20).to(args.device)

    # Optimizer
    optimizer = Adam(
        params=model.classifier.parameters(),   # Optimize only the classifier layer
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Meters
    meter_loss = AverageMeter()
    meter_accuracy = AverageMeter()
    train_accuracy, train_loss, val_accuracy, val_loss = 0,0,0,0

    epoch_bar = tqdm.trange(args.number_epochs, desc='Epoch')
    for epoch in epoch_bar:
        epoch_start_time = time()

        # Training
        model.train()
        torch.set_grad_enabled(True)
        batch_bar = tqdm.tqdm(data_loader_tr, desc='Batch')
        meter_loss.reset()
        meter_accuracy.reset()
        for batch in batch_bar:
            input_batch = batch[0].to(args.device)
            target = batch[1].to(args.device)
            logits = model(input_batch)

            number_samples = target.shape[0]
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == target).float().sum()/number_samples
            loss = F.cross_entropy(logits, target)
            meter_accuracy.update(accuracy, number_samples)
            meter_loss.update(loss, number_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # batch_bar.set_postfix({'loss': loss.item()})

        train_accuracy, train_loss = meter_accuracy.get_average(), meter_loss.get_average()
        epoch_bar.set_postfix({"loss": train_loss,
                               "accuracy": train_accuracy})
        writer.add_scalar("/train/loss", train_loss, epoch)
        writer.add_scalar("/train/accuracy", train_accuracy, epoch)

        # Validation
        model.eval()
        torch.set_grad_enabled(False)
        batch_bar = tqdm.tqdm(data_loader_val, desc='Batch')
        meter_loss.reset()
        meter_accuracy.reset()
        for batch in batch_bar:
            input_batch = batch[0].to(args.device)
            target = batch[1].to(args.device)
            logits = model(input_batch)

            number_samples = target.shape[0]
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == target).float().sum()/number_samples
            loss = F.cross_entropy(logits, target)
            meter_accuracy.update(accuracy, number_samples)
            meter_loss.update(loss, number_samples)

        val_accuracy, val_loss = meter_accuracy.get_average(), meter_loss.get_average()
        epoch_time = time()-epoch_start_time

        epoch_bar.set_postfix({"loss": val_loss,
                               "accuracy": val_accuracy})
        writer.add_scalar("/validation/loss", val_loss, epoch)
        writer.add_scalar("/validation/accuracy", val_accuracy, epoch)
        writer.add_scalar("time_per_epoch", epoch_time, epoch)

    torch.save(model.classifier.state_dict(), str(args.logdir / "final_model.pt"))
    return {"train": {"accuracy": train_accuracy, "loss": train_loss},
            "validation": {"accuracy": val_accuracy, "loss": val_loss}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--runpath", type=Path, required=True)
    parser.add_argument("--datapath", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--number_epochs", type=int, required=True)
    parser.add_argument("--number_workers", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random_hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    args.logdir = args.runpath / f"bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_{random_hash}"

    metric_dictionary = train(args)

    config = vars(args)
    final_log_dictionary = {"config": config,
                            "results": metric_dictionary}
    with open(args.logdir/"final_results.yaml", "w") as outfile:
        pyaml.dump(final_log_dictionary, outfile)
