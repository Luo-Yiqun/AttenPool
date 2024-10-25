# Yiqun Luo (luo2@andrew.cmu.edu)

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Subset
try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    print("Intel Extension for PyTorch is not available. Continuing without optimization.")
    print(e)
else:
    print("Intel Extension for PyTorch optimization applied successfully.")

from torchsummaryX import summary
from torch_geometric.loader import DataLoader

from megnet_pt import MEGNet, POOLING_LIST
import tqdm
import random
import numpy as np
import argparse
import os

if "ipex" in locals():
    device = 'xpu' if torch.xpu.is_available() else 'cpu'
else:
    device = "cpu"
print("Device:", device)

random.seed(11)
torch.manual_seed(11)
Ha_to_eV = 27.211386245981


def train_epoch(
        model,
        criterion,
        optimizer,
        dataloader, 
        ) -> float:
    losses = []
    idx = 0
    model.train()
    for batch in dataloader:
        if idx % 1000 == 0:
            print(f"{idx}, loss: {np.nanmean(losses[-1000:])}")
        idx += 1

        batch.to(device)
        outputs = model(batch)
        loss = criterion(outputs, batch['y'])
        losses.append(loss.item())
        # if torch.isnan(loss).any():
        #    print(idx, loss)
        #    print(batch.x, batch.edge_index, batch.edge_attr, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.nanmean(losses)


def evaluate(model, criterion, dataloader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch['y'])
            losses.append(loss.item())
    loss = np.nanmean(losses)
    print(f"loss: {loss}")
    return loss


def get_RMSE(model, dataloader) -> float:
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch['y'])
            losses.append(loss.item())
    RMSE = np.sqrt(np.nanmean(losses))
    print(f"RMSE: {RMSE}")
    return RMSE


def get_MAE(model, dataloader) -> float:
    criterion = nn.L1Loss()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch['y'])
            losses.append(loss.item())
    MAE = np.nanmean(losses) * Ha_to_eV
    print(f"MAE: {MAE}eV")
    return MAE


def main():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--dataset", default = "../MEGNet/QM9MEGNet.pt", help = "Dataset")
    parser.add_argument("--pooling", type=str, default="Set2Set", choices=POOLING_LIST, help="Pooling method to use")
    parser.add_argument('-l', "--load", action = "store_true", help = "Whether to load model")
    parser.add_argument("--model_prefix", type = str, default = "QM9MEGNet", help = "path to the model")
    parser.add_argument('-n', "--nepochs", type = int, default = 2, help = "Number of total epochs to run")

    # Parse the arguments
    args = parser.parse_args()
    model_path = f"{args.model_prefix}.pth"
    best_model_path = f"{args.model_prefix}_best.pth"
    
    dataset = torch.load(args.dataset)
    length = len(dataset)

    permutation = list(range(len(dataset)))
    random.shuffle(permutation)
    train_loader = DataLoader(Subset(dataset, permutation[: int(0.8 * length)]))
    valid_loader = DataLoader(Subset(dataset, permutation[int(0.8 * length): int(0.9 * length)]))
    test_loader = DataLoader(Subset(dataset, permutation[int(0.9 * length):]))

    model = MEGNet(
        n_node_features = 27,
        n_edge_features = 27,
        n_global_features = 2,
        pooling = args.pooling
    ).to(device)

    for _, data in enumerate(train_loader):
        input = data
    summary(model, input.to(device))

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    if 'ipex' in locals():
        model, optimizer = ipex.optimize(model, optimizer = optimizer)

    if args.load:
        assert os.path.exists(model_path), f"No pre-trained model at {model_path}!"
        print("Loading model...")
        checkpoint = torch.load(model_path)
        nepochs_prev = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses, val_losses = checkpoint["train_losses"], checkpoint["val_losses"]
        train_MAE, val_MAE, test_MAE = checkpoint["train_MAE"], checkpoint["val_MAE"], checkpoint["test_MAE"]
    else:
        print("Initializing model...")
        nepochs_prev = 0
        train_losses, val_losses = [], []
        train_MAE, val_MAE, test_MAE = [], [], []
    
    if os.path.exists(best_model_path):
        print("Loading best model val MAE...")
        lowest_val_MAE = torch.load(best_model_path)["val_MAE"]
    else:
        lowest_val_MAE = 0

    for epoch in tqdm.trange(nepochs_prev + 1, args.nepochs + 1):
        train_losses.append(train_epoch(model, criterion, optimizer, train_loader))
        val_losses.append(evaluate(model, criterion, train_loader))
        train_MAE.append(get_MAE(model, train_loader))
        val_MAE.append(get_MAE(model, valid_loader))
        test_MAE.append(get_MAE(model, test_loader))

        torch.save({
            "pooling": args.pooling,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_MAE": train_MAE,
            "val_MAE": val_MAE,
            "test_MAE": test_MAE
            }, args.model_path)
        
        if val_MAE[-1] < lowest_val_MAE:
            lowest_val_MAE = val_MAE[-1]
            torch.save({
                "pooling": args.pooling,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_MAE": train_MAE,
                "val_MAE": val_MAE,
                "test_MAE": test_MAE
                }, args.model_path)

    return

if __name__ == "__main__":
    main()
