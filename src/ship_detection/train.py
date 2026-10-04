import torch
import csv
from ship_detection.evaluate import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy




def run_training_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        checkpoint_path,
        history_path
):
    training_history = []

    best_val_f1 = -float("inf")
    best_epoch = 0
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            _,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                },
                checkpoint_path
            )

            print(
                f"Saved new best model at epoch {best_epoch} "
                f"(F1: {best_val_f1:.4f})"
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2%} | "
            f"Precision: {val_precision:.2%} | "
            f"Recall: {val_recall:.2%} | "
            f"F1: {val_f1:.4f}"
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }
        )

    with history_path.open("w", newline="") as file:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(training_history)
    print(f"Saved training history to: {history_path}")
    
    return training_history