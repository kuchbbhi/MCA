# noc_cnn_lib.py

import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# 1. XY ROUTING LABEL LOGIC
# ============================

def xy_routing_next_direction(
    current: Tuple[int, int],
    dest: Tuple[int, int]
) -> int:
    """
    Compute next direction using simple deterministic XY routing.
    Returns class index:
        0 - local (deliver)
        1 - north
        2 - south
        3 - east
        4 - west
    """
    cx, cy = current
    dx, dy = dest

    if (cx, cy) == (dx, dy):
        return 0  # local / deliver

    # Move in X dimension first (E/W), then Y (N/S)
    if cx < dx:
        return 3  # east
    elif cx > dx:
        return 4  # west
    elif cy < dy:
        return 2  # south
    elif cy > dy:
        return 1  # north

    return 0


# ============================
# 2. SYNTHETIC DATASET
# ============================

def generate_single_sample(noc_size: int) -> Tuple[np.ndarray, int]:
    """
    Generate one synthetic training sample for the NoC.

    Returns:
        input_tensor: shape (C, H, W) as numpy array
        label: integer in [0..4] indicating direction
    """
    H = W = noc_size

    # Randomly choose src, dest, and current router positions
    src_x, src_y = random.randrange(H), random.randrange(W)
    dst_x, dst_y = random.randrange(H), random.randrange(W)
    cur_x, cur_y = random.randrange(H), random.randrange(W)

    # Random load map
    load_map = np.random.rand(H, W).astype(np.float32)

    src_map = np.zeros((H, W), dtype=np.float32)
    dst_map = np.zeros((H, W), dtype=np.float32)
    cur_map = np.zeros((H, W), dtype=np.float32)

    src_map[src_x, src_y] = 1.0
    dst_map[dst_x, dst_y] = 1.0
    cur_map[cur_x, cur_y] = 1.0

    input_tensor = np.stack([load_map, src_map, dst_map, cur_map], axis=0)

    label = xy_routing_next_direction((cur_x, cur_y), (dst_x, dst_y))

    return input_tensor, label


class NoCRoutingDataset(Dataset):
    """PyTorch dataset for NoC routing samples."""

    def __init__(self, num_samples: int, noc_size: int):
        super().__init__()
        self.inputs = []
        self.labels = []

        for _ in range(num_samples):
            x, y = generate_single_sample(noc_size)
            self.inputs.append(x)
            self.labels.append(y)

        self.inputs = np.stack(self.inputs, axis=0)  # (N, C, H, W)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)


# ============================
# 3. CNN MODEL
# ============================

class NoCRouterCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================
# 4. TRAIN / EVAL HELPERS
# ============================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================
# 5. PATH PREDICTION UTILITIES
# ============================

def direction_to_delta(direction: int) -> Tuple[int, int]:
    """
    Convert class index to coordinate delta.
    0 - local (0,0)
    1 - north (x-1, y)
    2 - south (x+1, y)
    3 - east  (x, y+1)
    4 - west  (x, y-1)
    """
    if direction == 1:      # north
        return -1, 0
    elif direction == 2:    # south
        return 1, 0
    elif direction == 3:    # east
        return 0, 1
    elif direction == 4:    # west
        return 0, -1
    else:                   # local
        return 0, 0


def build_input_tensor_for_state(
    noc_size: int,
    load_map: np.ndarray,
    src: Tuple[int, int],
    dst: Tuple[int, int],
    current: Tuple[int, int]
) -> torch.Tensor:
    """
    Build a single input tensor (C, H, W) for a given NoC state and packet info.
    """
    H = W = noc_size
    src_map = np.zeros((H, W), dtype=np.float32)
    dst_map = np.zeros((H, W), dtype=np.float32)
    cur_map = np.zeros((H, W), dtype=np.float32)

    src_map[src[0], src[1]] = 1.0
    dst_map[dst[0], dst[1]] = 1.0
    cur_map[current[0], current[1]] = 1.0

    input_tensor = np.stack([load_map, src_map, dst_map, cur_map], axis=0)
    x = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    return x


def predict_path(
    model: nn.Module,
    noc_size: int,
    src: Tuple[int, int],
    dst: Tuple[int, int],
    max_hops: int = 16
) -> List[Tuple[int, int]]:
    """
    Use the trained model to predict a hop-by-hop path from src to dst.
    Stops when dst is reached or max_hops is exceeded.
    """
    model.eval()
    H = W = noc_size

    # For simplicity, use a random static load map
    load_map = np.random.rand(H, W).astype(np.float32)

    current = src
    path = [current]

    with torch.no_grad():
        for _ in range(max_hops):
            if current == dst:
                break

            x = build_input_tensor_for_state(noc_size, load_map, src, dst, current)
            outputs = model(x)
            _, predicted = outputs.max(1)
            direction = predicted.item()

            dx, dy = direction_to_delta(direction)
            new_x = current[0] + dx
            new_y = current[1] + dy

            if 0 <= new_x < H and 0 <= new_y < W:
                current = (new_x, new_y)
            else:
                break

            path.append(current)

    return path
