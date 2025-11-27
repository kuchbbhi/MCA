from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from noc_cnn_lib import (
    DEVICE,
    NoCRoutingDataset,
    NoCRouterCNN,
    train_one_epoch,
    evaluate,
    predict_path,
)

NOC_SIZE = 4          # 4x4 mesh
NUM_CHANNELS = 4      # load, src_map, dst_map, current_router_map
NUM_CLASSES = 5       # [local, N, S, E, W]

NUM_SAMPLES = 5000
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
MAX_HOPS = 16


def main():
    # 1) Dataset and loaders
    dataset = NoCRoutingDataset(NUM_SAMPLES, NOC_SIZE)

    split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2) Model, loss, optimizer
    model = NoCRouterCNN(
        in_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # 4) Demo path prediction
    src = (0, 0)
    dst = (3, 3)
    path = predict_path(model, NOC_SIZE, src, dst, MAX_HOPS)

    print(f"\nPredicted path from {src} to {dst}:")
    for step, coord in enumerate(path):
        print(f"Hop {step}: {coord}")


if __name__ == "__main__":
    main()
