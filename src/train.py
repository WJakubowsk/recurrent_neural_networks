import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import Transformer
from dataset import AudioDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(root_dir: str):
    dataset = AudioDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


def train_model(data_loader, n_tokens, d_model, n_head, d_hid, n_layers, dropout):
    # Train the model
    # Initialize model, loss function, and optimizer
    model = Transformer(n_tokens, d_model, n_head, d_hid, n_layers, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def evaluate_model(model, data_loader):
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main(args):
    root_dir = args.root_dir
    data_loader = load_data(root_dir)
    train_model(
        n_tokens=10,
        d_model=512,
        n_head=8,
        d_hid=2048,
        n_layers=6,
        dropout=0.1,
        data_loader=data_loader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/wiktor/studia/sem_8/DL/recurrent_neural_networks/data/train/audio",
    )
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-hid", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
