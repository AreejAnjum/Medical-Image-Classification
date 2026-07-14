
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
from evaluate import evaluate_model
from trainer import Trainer
from utils import plot_losses


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def start_memory_measurement(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()
    synchronize_device(device)


def get_memory_usage_mb(device):
    synchronize_device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2), "cuda_peak_allocated"
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2), "mps_current_allocated"
    return None, "not_available_on_cpu"


def get_data_path(config):
    data_path = Path(config["DATA_PATH"])
    if data_path.is_absolute():
        return data_path
    return Path(__file__).resolve().parent.parent / data_path


def run_experiment(model,config, model_name,data_name, data_augmentation, device):

    train_loader, val_loader, test_loader = get_loaders(data_name, data_path=get_data_path(config), batch_size=config["BATCH_SIZE"], val_split=config.get("VAL_SPLIT"), seed=config.get("SEED", 42), transform=data_augmentation) 
        
    num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["LEARNING_RATE"])
    trainer = Trainer(model, criterion, optimizer, device)
    start_memory_measurement(device)
    start_time = time.perf_counter()
    
    train_losses, val_losses = trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"], patience=config["PATIENCE"])      
    peak_train_memory, train_memory_type = get_memory_usage_mb(device)
           
    training_time = time.perf_counter() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    if peak_train_memory is not None:
        print(f"Training memory ({train_memory_type}): {peak_train_memory:.2f} MB")
    else:
        print("Training memory: not available on CPU")

    start_memory_measurement(device)
    inference_start_time = time.perf_counter()

    accuracy, precision, recall, macro_f1 = evaluate_model(model, test_loader, device)

    peak_inference_memory, inference_memory_type = get_memory_usage_mb(device)
    inference_time = time.perf_counter() - inference_start_time
    print(f"Total inference time: {inference_time:.2f} seconds")
    if peak_inference_memory is not None:
        print(f"Inference memory ({inference_memory_type}): {peak_inference_memory:.2f} MB")
    else:
        print("Inference memory: not available on CPU")

    output_dir = Path(config.get("OUTPUT_DIR", "results"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)




    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}_{data_name}.pt"

    torch.save(model.state_dict(), model_path)
    plot_losses({"losses": (train_losses, val_losses)}, title=f"Training and Validation loss result for {model_name} on {data_name}", out_path = output_dir)

    test_metrics = {
        "num_params": num_params,
        "total_params": total_params,
        "training_time": training_time,
        "peak_train_memory": peak_train_memory,
        "train_memory_type": train_memory_type,
        "inference_time": inference_time,
        "inference_latency_per_sample": inference_time/len(test_loader.dataset),
        "peak_inference_memory": peak_inference_memory, 
        "inference_memory_type": inference_memory_type,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1 
    }


    return test_metrics
