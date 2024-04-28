import os
import torch
import torchaudio
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_wave(waveform, max_length=16000) -> torch.Tensor:
    """
    Preprocesses a waveform by performing feature extraction and normalization.

    Args:
        waveform (Tensor): Input waveform tensor loaded using torchaudio.load().
        max_length (int): Maximum length of the waveform. Waveforms will be zero-padded to this length.

    Returns:
        Tensor: Preprocessed feature tensor.
    """
    waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)))
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / std

    return waveform


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.true_categories = sorted(os.listdir(root_dir))
        self.classes = sorted(
            [
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
                "unknown",
            ]
        )  # silence

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        class_name = self.true_categories[idx]
        class_dir = os.path.join(self.root_dir, class_name)
        files = os.listdir(class_dir)
        file = files[0]  # Assuming each class has at least one file
        file_path = os.path.join(class_dir, file)
        waveform, _ = torchaudio.load(file_path)
        waveform = preprocess_wave(waveform)

        # return idx if it is in the classes else return the index of unknown
        if class_name in self.classes:
            return waveform, self.classes.index(class_name)
        else:
            return waveform, self.classes.index("unknown")
