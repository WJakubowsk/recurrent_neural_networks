import os
import numpy as np
import argparse
import torch
from datasets import load_dataset
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def preprocess_wave(waveform: np.array, target_sr: int = 20, max_length: int = 32) -> torch.Tensor:
    """
    Preprocesses a waveform by performing feature extraction, downsampling, and normalization.

    Args:
        waveform (np.array): Input waveform numpy array.
        target_sr (int): Target sampling rate after downsampling.

    Returns:
        Tensor: Preprocessed feature tensor.
    """
    # fill the waveform with zeros if it has nan values
    missing_indices = np.isnan(waveform)
    waveform[missing_indices] = 0
    
    # Perform MFCC feature extraction
    waveform_mfcc = librosa.feature.mfcc(y=waveform, sr=waveform.shape[0], n_mfcc=target_sr)

    # Pad or truncate the MFCC sequences to a fixed length
    if waveform_mfcc.shape[1] < max_length:
        waveform_mfcc = np.pad(waveform_mfcc, ((0, 0), (0, max_length - waveform_mfcc.shape[1])), mode='constant', constant_values=0)
    else:
        waveform_mfcc = waveform_mfcc[:, :max_length]
    
    # Convert to PyTorch tensor
    torch_waveform = torch.tensor(waveform_mfcc, dtype=torch.float32)
    
    # Normalize the waveform
    mean = torch_waveform.mean()
    std = torch_waveform.std()
    torch_waveform = (torch_waveform - mean) / std
    
    return torch_waveform.view(-1)

def get_datasets(version: str = "v0.01", sr: int = 8000, max_length: int = 16000):
    """
    Unpacks the dataset into train, validation, and test splits of desired version.
    Args:
        version (str): Version of the dataset.
        sr (int): Target sampling rate for the audio.
    """
    dataset = load_dataset("speech_commands", version)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    datasets = [train_dataset, val_dataset, test_dataset]
    new_datasets = []
    for dataset in datasets:
        dataset = dataset.map(lambda x: {"audio": preprocess_wave(x["audio"]['array'], target_sr=sr, max_length=max_length), "label": x["label"] if x["label"] < 10 else 11 if x["label"] == 30 else 10})
        dataset.set_format(type="torch", columns=["audio", "label"])
        new_datasets.append(dataset)
    return new_datasets

def main(args):
    train, val, test = get_datasets(args.version, args.sr)
    print(f"Number of training samples: {len(train)}")
    print(f"Number of validation samples: {len(val)}")
    print(f"Number of test samples: {len(test)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v0.01")
    parser.add_argument("--sr", type=int, default=400)
    args = parser.parse_args()
    main(args)