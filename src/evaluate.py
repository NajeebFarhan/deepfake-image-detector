import torchmetrics
import torch
from model import deepfake_model
from dataset import fetch_deepfake_images, dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deepfake_model()
    model.load_state_dict(torch.load("deepfake_model.pth"))
    model.to(device)

    training_dataset, testing_dataset, validation_dataset = fetch_deepfake_images()

    validation_loader = dataloader(validation_dataset)

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

    accuracy.reset()

    model.eval()
    with torch.no_grad():
        for X, y in validation_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            accuracy.update(logits, y)

    result_accuracy = accuracy.compute()

    print(result_accuracy.item())