import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from models import Transformer, LSTM
from dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int = 32, sampling_rate: int = 8000, max_length: int = 16000, task: str = "main") -> tuple:
    """
    Load the dataset of the desired version and return the dataloaders for every split.
    Args:
        version (str): Version of the dataset.
        batch_size (int): Batch size for the dataloaders.
        sampling_rate (int): Sampling rate for the audio data.
    """
    print("loading data...")
    if task == "main":
        train, val, test = get_main_task_datasets(sr=sampling_rate, max_length=max_length)
    elif task == "silence":
        train, val, test = get_silence_datasets(sr=sampling_rate, max_length=max_length)

    elif task == "unknown":
        train, val, test = get_unknown_datasets(sr=sampling_rate, max_length=max_length)
    else:
        train, val, test = get_datasets(sr=sampling_rate, max_length=max_length)       
    
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def train_model(model_class, model_params, train_loader, val_loader, model_filename, num_epochs=10):
    torch.manual_seed(123)
    print("training models...")
    # Initialize model, loss function, and optimizer
    model = model_class(**model_params)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(train_loader):
            if args.model_class == "Transformer":
                inputs, labels = data["audio"].to(device), data["label"].to(device)
            else:
                inputs, labels = data["audio"].unsqueeze(1).to(device), data["label"].to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients to prevent them from becoming too large
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # You can adjust max_norm as needed
            
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 10 == 9:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))

        # Calculate training accuracy after each epoch
        train_accuracy = round(100 * correct_train / total_train, 2)
        print(f"Accuracy on train (%) after epoch {epoch + 1}: {train_accuracy}")
        # save train accuracy to txt file
        with open(f"results/separate/{model_filename}_train_accuracy.txt", "a") as f:
            f.write(f"{train_accuracy}\n")

        # Evaluate on validation set
        val_accuracy = round(evaluate_model(model, val_loader) * 100, 2)
        print(f"Accuracy on validation (%) after epoch {epoch + 1}: {val_accuracy}")
        # save val accuracy to txt file
        with open(f"results/separate/{model_filename}_val_accuracy.txt", "a") as f:
            f.write(f"{val_accuracy}\n")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, "./pretrained/separate/" + model_filename + ".pth")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)


def evaluate_model(model, data_loader):
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            if args.model_class == "Transformer":
                inputs, labels = data["audio"].to(device), data["label"].to(device)
            else:
                inputs, labels = data["audio"].unsqueeze(1).to(device), data["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main(args):
    train_loader, val_loader, test_loader = load_data(args.batch_size, args.max_length, args.sampling_rate, task=args.task)

    if args.model_class == "Transformer":
        model = Transformer
        model_params = {
            "input_size": args.max_length * args.sampling_rate,
            "num_classes": args.output_size,
            "n_head": args.n_head,
            "num_encoder_layers": args.n_encoder_layers,
            "num_decoder_layers": args.n_decoder_layers,
            "dim_feedforward": args.d_hid,
            "dropout": args.dropout,
        }
        model_filename = f"{args.task}_{args.model_class}_input_size-{args.max_length * args.sampling_rate}_n_head-{args.n_head}_n_encoder_layers-{args.n_encoder_layers}_n_decoder_layers-{args.n_decoder_layers}_d_hid-{args.d_hid}_dropout-{args.dropout}"
    elif args.model_class == "LSTM":
        model = LSTM
        model_params = {
            "input_size": args.max_length * args.sampling_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            # "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "output_size": args.output_size,
        }
        model_filename = f"{args.task}_{args.model_class}_input_size-{args.max_length * args.sampling_rate}_hidden_size-{args.hidden_size}_layers-{args.num_layers}_bidirectional-{args.bidirectional}"
    else:
        raise ValueError("Unsupported model type")
    trained_model = train_model(model, model_params, train_loader, val_loader, model_filename, args.num_epochs)
    test_accuracy = round(evaluate_model(trained_model, test_loader) * 100, 2)
    print(f"Accuracy on test (%): {test_accuracy}")

    # get confusion matrix
    confusion_matrix = torch.zeros(args.output_size, args.output_size)
    with torch.no_grad():
        for data in test_loader:
            if args.model_class == "Transformer":
                inputs, labels = data["audio"].to(device), data["label"].to(device)
            else:
                inputs, labels = data["audio"].unsqueeze(1).to(device), data["label"].to(device)    
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{model_filename}.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["main", "silence", "unknown"], default="main")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--model-class", type=str, choices=["Transformer", "LSTM"], default="LSTM")
    parser.add_argument("--sampling-rate", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--output-size", type=int, default=10)
    parser.add_argument("--d-hid", type=int, default=128)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)
