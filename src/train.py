import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from models import Transformer, LSTM
from dataset import get_datasets
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(version: str = "v0.01", batch_size: int = 32, sampling_rate: int = 8000, max_length: int = 16000) -> tuple:
    """
    Load the dataset of the desired version and return the dataloaders for every split.
    Args:
        version (str): Version of the dataset.
        batch_size (int): Batch size for the dataloaders.
        sampling_rate (int): Sampling rate for the audio data.
    """
    print("loading data...")
    train, val, test = get_datasets(version, sr=sampling_rate, max_length=max_length)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def train_model(model_class, model_params, train_loader, val_loader, model_filename):
    torch.manual_seed(123)
    print("training model...")
    # Initialize model, loss function, and optimizer
    model = model_class(**model_params)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data["audio"].to(device), data["label"].to(device)
            # print("inputs", inputs.shape)
            # print("labels", labels.shape)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            # print("outputs", outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 10 == 9:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        # Calculate training accuracy after each epoch
        train_accuracy = round(100 * correct_train / total_train, 2)
        print(f"Accuracy on train (%) after epoch {epoch + 1}: {train_accuracy}")

        # Evaluate on validation set
        val_accuracy = round(evaluate_model(model, val_loader) * 100, 2)
        print(f"Accuracy on validation (%) after epoch {epoch + 1}: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, "./pretrained/" + model_filename + ".pth")
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def evaluate_model(model, data_loader):
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data["audio"].to(device), data["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main(args):
    train_loader, val_loader, test_loader = load_data(args.version, args.batch_size, args.max_length, args.sampling_rate)

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
        model_filename = f"{args.model_class}_input_size-{args.max_length * args.sampling_rate}_n_head-{args.n_head}_n_encoder_layers-{args.n_encoder_layers}_n_decoder_layers-{args.n_decoder_layers}_d_hid-{args.d_hid}_dropout-{args.dropout}"
    elif args.model_class == "LSTM":
        model = LSTM
        model_params = {
            "input_size": args.input_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        }
        model_filename = f"{args.model_class}_hidden_size-{args.hidden_size}_layers-{args.num_layers}_dropout-{args.dropout}_bidirectional-{args.bidirectional}"
    else:
        raise ValueError("Unsupported model type")
    trained_model = train_model(model, model_params, train_loader, val_loader, model_filename)
    test_accuracy = round(evaluate_model(trained_model, test_loader) * 100, 2)
    print(f"Accuracy on test (%): {test_accuracy}")

    # get confusion matrix
    confusion_matrix = torch.zeros(args.output_size, args.output_size)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data["audio"].to(device), data["label"].to(device)
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
    parser.add_argument("--model-class", type=str, choices=["Transformer", "LSTM"], default="Transformer")
    parser.add_argument("--version", type=str, default="v0.01")
    parser.add_argument("--sampling-rate", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--output-size", type=int, default=12)
    parser.add_argument("--d-hid", type=int, default=128)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=16000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)
