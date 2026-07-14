from pathlib import Path
import torch
import torch.nn as nn
import models



def build_pretrained_model(task3_config, mode, device):
    model_name = task3_config["MODEL"]
    model_class = getattr(models, model_name)
  
    if mode == "scratch":
        return model_class(
            in_channels=task3_config["channels"],
            num_classes=task3_config["num_classes"],
        ).to(device)
    
    # pretrained / finetune
    model = model_class(
        in_channels=task3_config["SOURCE_DATA"]["channels"],
        num_classes=task3_config["SOURCE_DATA"]["num_classes"],
    ).to(device)



    checkpoint_path = Path(task3_config["CHECKPOINT"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing source checkpoint: {checkpoint_path}. "
            "Run task2 first so the Light_ResNet18 orgs checkpoint is created."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    #adjust classifier for new number of classes
    if isinstance(model.classifier, nn.Sequential):# VGG / AlexNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, task3_config["num_classes"]).to(device)
    else:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, task3_config["num_classes"]).to(device)

    # freeze parameters 
    for p in model.parameters():
        p.requires_grad = False

    #unfreeze classifier parameters
    for p in model.classifier.parameters():
        p.requires_grad = True

    if mode == "fine_tune":
        if "ResNet" in model_name:
            for p in model.stage4.parameters():
                p.requires_grad = True
        elif "VGG16" in model_name:
            for p in model.features[-1].parameters():
                p.requires_grad = True
                
        elif "AlexNet" in model_name:
            for layer in [model.features[12]]:
                for p in layer.parameters():
                    p.requires_grad = True
    return model
