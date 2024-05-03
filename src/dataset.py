import os
import numpy as np
import argparse
import torch
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_wave(waveform: np.array, max_length: int = 16000) -> torch.Tensor:
    """
    Preprocesses a waveform by performing feature extraction and normalization.

    Args:
        waveform (np.array): Input waveform numpy arrray.
        max_length (int): Maximum length of the waveform. Waveforms will be zero-padded to this length.

    Returns:
        Tensor: Preprocessed feature tensor.
    """
    torch_waveform = torch.tensor(waveform, dtype=torch.float32)
    torch_waveform = torch.nn.functional.pad(torch_waveform, (0, max_length - torch_waveform.size(0)))
    mean = torch_waveform.mean()
    std = torch_waveform.std()
    torch_waveform = (torch_waveform - mean) / std
    return torch_waveform

def get_datasets(version: str = "v0.01"):
    """
    Unpacks the dataset into train, validation, and test splits of desired version.
    """
    dataset = load_dataset("speech_commands", version)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    datasets = [train_dataset, val_dataset, test_dataset]
    new_datasets = []
    for dataset in datasets:
        dataset = dataset.map(lambda x: {"audio": preprocess_wave(x["audio"]['array']), "label": 10 if x["label"] >= 10 else x["label"]})
        dataset.set_format(type="torch", columns=["audio", "label"])
        new_datasets.append(dataset)
    return new_datasets

def main(args):
    train, val, test = get_datasets(args.version)
    print(f"Number of training samples: {len(train)}")
    print(f"Number of validation samples: {len(val)}")
    print(f"Number of test samples: {len(test)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v0.01")
    args = parser.parse_args()
    main(args)