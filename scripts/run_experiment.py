import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

from src.models.KANModel import KANNet
from src.models.MLPModel import MLPNet
from src.train_test import train_and_test_models
from src.utils import process_single_file

from osgeo import gdal

# ======================
# CONFIG
# ======================

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = 'data/train_validation_data'
TEST_DIR = 'data/test_data'

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

MLP_MODEL_PATH = 'checkpoints/pretrained/MLPmodel.pth'
KAN_MODEL_PATH = 'checkpoints/pretrained/KANmodel.pth'


# ======================
# DATASET CLASS
# ======================

class dataset_class(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx].item()


# ======================
# TRAIN DATA LOADING
# ======================

def load_train_data():

    gdal.PushErrorHandler('CPLQuietErrorHandler')

    all_data = []
    all_labels = []

    print("Loading TRAIN data...")

    for i in range(1, 4):
        for j in range(1, 10):
            file_path = os.path.join(TRAIN_DIR, f'c{i}_r{j}.tif')
            X, Y = process_single_file(file_path, i - 1)
            all_data.append(X)
            all_labels.append(Y)

    X = np.vstack(all_data)
    Y = np.vstack(all_labels)

    combined = np.hstack((X, Y))
    np.random.shuffle(combined)

    X_shuffled = combined[:, :-1]
    Y_shuffled = combined[:, -1:]

    mean = np.mean(X_shuffled, axis=0)
    std = np.std(X_shuffled, axis=0)
    X_normalized = (X_shuffled - mean) / std

    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_shuffled, dtype=torch.long).squeeze()

    return X_tensor, Y_tensor, mean, std


# ======================
# TEST DATA LOADING
# ======================

def load_test_data(mean, std):

    all_data = []
    all_labels = []

    print("Loading TEST data...")

    for i in range(1, 4):
        for j in range(1, 10):
            file_path = os.path.join(TEST_DIR, f'c{i}_r{j}.tif')
            X, Y = process_single_file(file_path, i - 1)
            all_data.append(X)
            all_labels.append(Y)

    X = np.vstack(all_data)
    Y = np.vstack(all_labels)

    combined = np.hstack((X, Y))
    np.random.shuffle(combined)

    X_shuffled = combined[:, :-1]
    Y_shuffled = combined[:, -1:]

    X_normalized = (X_shuffled - mean) / std

    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_shuffled, dtype=torch.long).squeeze()

    return X_tensor, Y_tensor


# ======================
# MAIN
# ======================

def main():

    print(f"Using device: {DEVICE}")

    # ----------------------
    # Load training data
    # ----------------------
    X_tensor, Y_tensor, mean, std = load_train_data()

    dataset = dataset_class(X_tensor, Y_tensor)

    train_idx, val_idx = train_test_split(range(len(X_tensor)), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------
    # Initialize models
    # ----------------------
    mlp_model = MLPNet().to(DEVICE)
    kan_model = KANNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # ----------------------
    # MLP
    # ----------------------
    optimizer = optim.Adam(mlp_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(MLP_MODEL_PATH):
        print(f"Loading MLP model from {MLP_MODEL_PATH}")
        mlp_model = torch.load(MLP_MODEL_PATH, map_location=DEVICE)
    else:
        results = train_and_test_models(
            mlp_model, DEVICE, train_loader, val_loader,
            optimizer, criterion, EPOCHS, scheduler,
            model_type=mlp_model, name='mlp'
        )
        torch.save(mlp_model, MLP_MODEL_PATH)

    # ----------------------
    # KAN
    # ----------------------
    optimizer = optim.Adam(kan_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(KAN_MODEL_PATH):
        print(f"Loading KAN model from {KAN_MODEL_PATH}")
        kan_model = torch.load(KAN_MODEL_PATH, map_location=DEVICE)
    else:
        results = train_and_test_models(
            kan_model, DEVICE, train_loader, val_loader,
            optimizer, criterion, EPOCHS, scheduler,
            model_type=kan_model, name='kan'
        )
        torch.save(kan_model, KAN_MODEL_PATH)

    # ----------------------
    # TEST DATA
    # ----------------------
    X_test, Y_test = load_test_data(mean, std)

    test_dataset = dataset_class(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\nEvaluating on TEST set...")

    from src.train_test import test

    for name, model in [("MLP", mlp_model), ("KAN", kan_model)]:
        loss, acc, precision, recall, f1 = test(model, DEVICE, test_loader, criterion)
        print(f"\n{name} Results:")
        print(f"Accuracy: {acc:.4%}")
        print(f"Precision: {precision:.5f}")
        print(f"Recall: {recall:.5f}")
        print(f"F1 Score: {f1:.5f}")


if __name__ == "__main__":
    main()
