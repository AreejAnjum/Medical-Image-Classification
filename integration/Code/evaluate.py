import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_loader, device):
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    label_list = []
    prediction_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.view(-1).long()
            outputs = model(images)
            prediction = outputs.argmax(dim=1)

            
            label_list.extend(labels.cpu().tolist())
            prediction_list.extend(prediction.cpu().tolist())

    accuracy = accuracy_score(label_list, prediction_list)
    precision = precision_score(label_list, prediction_list, average="macro", zero_division=0)
    recall = recall_score(label_list, prediction_list, average="macro", zero_division=0)
    macro_f1 = f1_score(label_list, prediction_list, average="macro", zero_division=0)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

  
    return accuracy, precision, recall, macro_f1
