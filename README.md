# Recurrent Neural Networks
Joint work with Zuzanna Kotlińska on the Audio Signal Processing project as part of the Deep Learning course at Warsaw University of Technology.

## Dataset

The data contains roughly 65000 examples of voice commands, out of which we distinguish from the following: *yes*, *no*, *up*, *down*,
*left*, *right*, *on*, *off*, *stop*, *go* and ***silence***. Other examples were labeled as *unkown*.
Dataset is availble on Kaggle under this [link](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) in the form of ZIP file as well as on HuggingFace under this [link](https://huggingface.co/datasets/speech_commands/viewer) in already preprocessed TorchDict form.

## Task

The aim of this project is to design and compare various Deep Learning architectures to tackle the problem of audio classification (mutliclass). The are two groups of architectures that were inspected: the first group relied on LSTM, whereas the latter - on Transformer approach.

## Instruction
Install all necessary libraries, specified in *requirements.txt*.

Download the dataset and place it in the *data/* dir in the root of the project.

To obtain the results, simply run one of the following Python scripts:
* `python train.py` - specify the variant of the architecture by providing model class ("LSTM", "Transformer") and parameters,
* `python train_separate.py` - as above, but use this for preparing committee models for ensemble model,
* `python ensemble.py` - make prediction on the ensemble of three separate DL architectures: for silence detection, for unknown sound detection, and for main classess classification (sequencial model).

