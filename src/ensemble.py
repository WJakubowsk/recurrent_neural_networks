import torch
import torch.nn as nn
from models import LSTM
from dataset import get_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

class EnsembleRNN(nn.Module):
    def __init__(self, main_task_model, unknown_model, silence_model):
        """
        Initialize the ensemble model with the main task model, unknown model, and silence model.
        """
        super(EnsembleRNN, self).__init__()
        self.main_task_model = main_task_model
        self.unknown_model = unknown_model
        self.silence_model = silence_model

    def forward(self, x):
        """
        Forward pass of the ensemble model. The input first goes through model classifying silence, then (if negative prediction)
        goes through model classifying unknown, and finally goes through the main task model.
        """
        # Silence model
        silence_output = self.silence_model.forward(x)
        silence_prediction = torch.argmax(silence_output, dim=1)
        silence_prediction = silence_prediction == 1

        # Unknown model
        unknown_output = self.unknown_model.forward(x)
        unknown_prediction = torch.argmax(unknown_output, dim=1)
        unknown_prediction = unknown_prediction == 1

        # Main task model
        main_task_output = self.main_task_model.forward(x)
        main_task_prediction = torch.argmax(main_task_output, dim=1)

        # Combine predictions
        prediction = torch.where(silence_prediction, silence_prediction * 11, torch.where(unknown_prediction, unknown_prediction * 10, main_task_prediction))
        return prediction
    
def main():
    main_mask_model_path = "./pretrained/separate/main_LSTM_input_size-320_hidden_size-128_layers-2_bidirectional-True.pth"
    silence_model_path = "./pretrained/separate/silence_LSTM_input_size-320_hidden_size-128_layers-2_bidirectional-True.pth"
    unkown_model_path = "./pretrained/separate/unknown_LSTM_input_size-320_hidden_size-128_layers-2_bidirectional-True.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = get_datasets(sr = 20, max_length = 16)

    main_task_model = LSTM(input_size=320, hidden_size=128, num_layers=2, output_size=10, bidirectional=True)
    main_task_model.load_state_dict(torch.load(main_mask_model_path))

    silence_model = LSTM(input_size=320, hidden_size=128, num_layers=2, output_size=2, bidirectional=True)
    silence_model.load_state_dict(torch.load(silence_model_path))

    unknown_model = LSTM(input_size=320, hidden_size=128, num_layers=2, output_size=2, bidirectional=True)
    unknown_model.load_state_dict(torch.load(unkown_model_path))

    ensemble_model = EnsembleRNN(main_task_model, unknown_model, silence_model)
    ensemble_model.to(device)

    correct = 0
    total = 0

    test_loader = DataLoader(test, batch_size=32, shuffle=True)

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data["audio"].unsqueeze(1).to(device), data["label"].to(device)
            outputs = ensemble_model.forward(inputs)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    
    print(f"Accuracy on test (%): {round(correct / total * 100, 2)}")

    # get confusion matrix
    confusion_matrix = torch.zeros(12, 12)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data["audio"].unsqueeze(1).to(device), data["label"].to(device)    
            outputs = ensemble_model(inputs)
            predicted = outputs
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/ensemble.png")
    


if __name__ == "__main__":
    main()