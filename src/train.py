def train(model, loader, optimizer, criterion, device, epochs=5):
    model.train()
    dataset_size = len(loader.dataset)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} -----------------------------------")
        for batch_idx, (X, y) in enumerate(loader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                print(f"Loss {loss:.4f} [{current}/{dataset_size}]")

    print("Finished Training")


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import deepfake_model
    from dataset import fetch_deepfake_images, dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset, testing_dataset, validation_dataset = fetch_deepfake_images()

    train_loader = dataloader(training_dataset, shuffle=True)

    deepfake_model = deepfake_model().to(device)

    optimizer = optim.Adam(deepfake_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(deepfake_model, train_loader, optimizer, criterion, device=device, epochs=5)

    torch.save(deepfake_model.state_dict(), "deepfake_model.pth")