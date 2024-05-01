import os
import argparse
import shutil

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    default="C:/Users/User/Documents/Deep_learning/recurrent_neural_networks/data/",
    help="Root directory containing audio data and list files"
)
args = parser.parse_args()

# Create directories for train, test, and validation data
audio_dir = os.path.join(args.root_dir, 'audio')
testing_list_path = os.path.join(args.root_dir, 'testing_list.txt')
validation_list_path = os.path.join(args.root_dir, 'validation_list.txt')
train_dir = os.path.join(args.root_dir, 'train')
test_dir = os.path.join(args.root_dir, 'test')
validation_dir = os.path.join(args.root_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Read paths from testing_list.txt and validation_list.txt
with open(testing_list_path, 'r') as f:
    test_paths = f.read().splitlines()
   
with open(validation_list_path, 'r') as f:
    validation_paths = f.read().splitlines()

# Move files to their respective directories
for root, _, files in os.walk(audio_dir):
    for file in files:
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, audio_dir)
        rel_path = rel_path.replace("\\", "/")
        dest_path = None
        if rel_path in test_paths:
            dest_path = os.path.join(test_dir, rel_path)
        elif rel_path in validation_paths:
            dest_path = os.path.join(validation_dir, rel_path)
        else:
            dest_path = os.path.join(train_dir, rel_path)
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(file_path, dest_path)