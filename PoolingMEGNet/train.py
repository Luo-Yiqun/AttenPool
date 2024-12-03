import torch
import gc
from torch import nn
import torch.optim as optim
from torch.utils.data import Subset
from torchsummaryX import summary
from torch_geometric.loader import DataLoader

from megnet_pt import MEGNet, POOLING_LIST
import tqdm
import random
import numpy as np
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--dataset", default = "QM9MEGNet.pt", help = "Dataset")
    parser.add_argument('-l', "--load", action = "store_true", help = "Whether to load model")
    parser.add_argument("--model_prefix", type = str, default = "QM9MEGNet", help = "path to the model")
    parser.add_argument("--model_path", type = str, default = "QM9MEGNet.pth", help = "path to the model")

    parser.add_argument("--n1", type = int, default = 64, help = "n1 number of hidden neurons")
    parser.add_argument("--n2", type = int, default = 32, help = "n2 number of hidden neurons between blocks")
    parser.add_argument("--n3", type = int, default = 16, help = "n3 number of hidden neurons in the last layer")
    parser.add_argument("--n_blocks", type = int, default = 2, help = "Number of GraphNetwork blocks to use in update")
    parser.add_argument("--pooling", type=str, default="Set2Set", choices=POOLING_LIST, help="Pooling method to use")
    parser.add_argument("--num_heads", type = int, default = 4, help = "Number of attention heads")
    parser.add_argument("--dropout", type = float, default = 0.1, help = "Dropout rate")

    parser.add_argument("--loss", type = str, default = "L1Loss", help = "Loss function")
    parser.add_argument("--optim", type = str, default = "Adam", help = "Optimizer")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument('-n', "--nepochs", type = int, default = 2, help = "Number of total epochs to run")
    parser.add_argument("--batch_size", type = int, default = 2, help = "Batch size")

    # Parse the arguments
    args = parser.parse_args()
    model_path = f"{args.model_prefix}.pth"
    best_model_path = f"{args.model_prefix}_best.pth"
    
    dataset = torch.load(args.dataset)
    length = len(dataset)

    permutation = list(range(len(dataset)))
    random.shuffle(permutation)
    train_loader = DataLoader(Subset(dataset, permutation[: int(0.8 * length)]), batch_size = args.batch_size)
    valid_loader = DataLoader(Subset(dataset, permutation[int(0.8 * length): int(0.9 * length)]), batch_size = args.batch_size)
    test_loader = DataLoader(Subset(dataset, permutation[int(0.9 * length):]), batch_size = args.batch_size)

    model = MEGNet(
        n_node_features = 27,
        n_edge_features = 27,
        n_global_features = 2,
        n1 = args.n1,
        n2 = args.n2,
        n3 = args.n3,
        n_blocks = args.n_blocks,
        pooling = args.pooling,
        num_heads = args.num_heads,
        dropout = args.dropout,
    ).to(device)

    for _, data in enumerate(train_loader):
        input = data
    summary(model, input.to(device))

    criterion = getattr(nn, args.loss)().to(device)
    optimizer = getattr(optim, args.optim)(model.parameters(), lr = args.lr)

    import wandb
    wandb.login(key="bc022b99e5a39b97fc6ae8c641ab328e9f52d2e6") 
    run = wandb.init(
        name=f"QM9_MEGNET_{args.pooling}",  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Almlows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="project",  ### Project should be created in your wandb account
    )

    torch.cuda.empty_cache()
    gc.collect()

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

        # torch.save({
        #     "pooling": args.pooling,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'epoch': epoch,
        #     "train_losses": train_losses,
        #     "val_losses": val_losses,
        #     "train_MAE": train_MAE,
        #     "val_MAE": val_MAE,
        #     "test_MAE": test_MAE
        #     }, model_path)
        
        wandb.log({
            "train_loss": train_losses[-1],
            'validation_loss': val_losses[-1],
            "train_MAE": train_MAE[-1],
            "val_MAE": val_MAE[-1],
            "test_MAE": test_MAE[-1]
        })
        
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
                }, model_path)
            
       
    run.finish()

    return

if __name__ == "__main__":
    main()
