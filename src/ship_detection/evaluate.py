import torch

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            true_positive += (
                (predictions == 1) & (labels == 1)
            ).sum().item()

            true_negative += (
                (predictions == 0) & (labels == 0)
            ).sum().item()

            false_positive += (
                (predictions == 1) & (labels == 0)
            ).sum().item()

            false_negative += (
                (predictions == 0) & (labels == 1)
            ).sum().item()
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall_denominator = true_positive + false_negative

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2 * (precision * recall) / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return (
        epoch_loss, 
        epoch_accuracy, 
        precision, 
        recall, 
        f1,
        true_positive,
        true_negative,
        false_positive,
        false_negative
    )