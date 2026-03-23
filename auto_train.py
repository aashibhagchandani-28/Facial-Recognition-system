import os
import pickle
import subprocess
import hashlib

DATASET_DIR = "dataset"
HASH_FILE = "dataset_hash.txt"
CLASSIFIER_FILE = "classifier.pkl"

MODEL_PATH = "../models/20180402-114759/20180402-114759.pb"


def get_dataset_hash():

    hash_md5 = hashlib.md5()

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in sorted(files):
            path = os.path.join(root, file)
            hash_md5.update(path.encode())

    return hash_md5.hexdigest()


def load_old_hash():

    if not os.path.exists(HASH_FILE):
        return None

    with open(HASH_FILE, "r") as f:
        return f.read()


def save_hash(h):

    with open(HASH_FILE, "w") as f:
        f.write(h)


def train_if_needed():

    new_hash = get_dataset_hash()
    old_hash = load_old_hash()

    if new_hash != old_hash or not os.path.exists(CLASSIFIER_FILE):

        print("New dataset detected. Training classifier...")

        cmd = [
            "python",
            "train_classifier.py",
            "--model-path", MODEL_PATH,
            "--input-dir", DATASET_DIR,
            "--classifier-path", CLASSIFIER_FILE,
            "--is-train",
            "--min-num-images-per-class", "2"
        ]

        subprocess.run(cmd)

        save_hash(new_hash)

        print("Training completed.")

    else:
        print("No new data. Skipping training.")


if __name__ == "__main__":

    train_if_needed()
