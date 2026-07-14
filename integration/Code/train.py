"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import argparse
import json
import random
from pathlib import Path
import torch
import models
from utils import write_csv
from runner import get_device, run_experiment
from transfer import build_pretrained_model


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     

def get_run_list(config, task, run_all=False):
    if task == "task1":
        if run_all:
            return [
                (data_name, model_name, None)
                for data_name in config["DATASETS"]
                for model_name in config["MODELS"]
            ]
        return [(config["DATA"], config["MODEL"], None)]

    if task == "task2":
        if run_all:
            return [
                (data_name, model_name, mode)
                for data_name in config["DATASETS"]
                for model_name in config["task2"]["MODELS"]
                for mode in config["task2"]["MODES"]
            ]
        return [(config["DATA"], config["MODEL"], "Baseline")]

    if task == "task3":
        return [
            (config["task3"]["TARGET_DATA"], config["task3"]["MODEL"], mode)
            for mode in config["task3"]["MODES"]
        ]

    raise ValueError(f"Unknown task: {task}")


def get_task2_model_name(model_name, mode):
    if mode == "Light":
        return f"Light_{model_name}"
    return model_name


def make_result_row(data_name, model_name, mode, config, result_metrics):
    return {
        "dataset": data_name,
        "model": model_name,
        "mode": mode or "Baseline",
        "epochs": config["EPOCHS"],
        "trainable_parameters": result_metrics["num_params"],
        "total_parameters": result_metrics["total_params"],
        "training_time_seconds": result_metrics["training_time"],
        "peak_training_memory_mb": result_metrics["peak_train_memory"],
        "training_memory_type": result_metrics["train_memory_type"],
        "accuracy": result_metrics["accuracy"] * 100,
        "precision": result_metrics["precision"] * 100,
        "recall": result_metrics["recall"] * 100,
        "macro_f1": result_metrics["macro_f1"] * 100,
        "inference_time_seconds": result_metrics["inference_time"],
        "peak_inference_memory_mb": result_metrics["peak_inference_memory"],
        "inference_memory_type": result_metrics["inference_memory_type"],
        "inference_latency_per_sample": result_metrics["inference_latency_per_sample"],
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",choices=["task1", "task2", "task3"],default="task1",)
    
    return parser.parse_args()
  
 
def main():
    args = parse_args()
    task = args.task
    config = load_config()    
    set_seed(config.get("SEED", 42))
    device = get_device()
    print(f"Training executing on device: {device}")
    run_all = config.get("RUN_ALL", False)
    rows = []
    experiment_list = get_run_list(config,task, run_all=run_all)
    data_augmentation = False

    for data_name, model_name, mode in experiment_list:
           
        if task == "task1": 
            model_class = getattr(models, model_name)
            model = model_class(in_channels=config["DATASETS"][data_name]["channels"], num_classes=config["DATASETS"][data_name]["num_classes"], drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", None)).to(device)
            
        elif task == "task2":
            model_name = get_task2_model_name(model_name, mode)
            model_class = getattr(models, model_name)
            model = model_class(in_channels=config["DATASETS"][data_name]["channels"], num_classes=config["DATASETS"][data_name]["num_classes"], drop_rate=config.get("DROP_RATE", 0.5), activation_str=config.get("ACTIVATION", None)).to(device)

        elif task == "task3" :
            model = build_pretrained_model(config["task3"], mode, device)
            model_name = f"{model_name}_{mode}" 
            data_augmentation = config["task3"].get("DATA_TRANSFORM", False)

        print(f"\nRunning {model_name} on {data_name}")     
        result_metrics = run_experiment(model, config, model_name, data_name, data_augmentation, device=device)

        rows.append(make_result_row(data_name, model_name, mode, config, result_metrics))

    output_dir = Path(config.get("OUTPUT_DIR", "results"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    write_csv(rows,output_path=output_dir / f"{task}_test_metrics.csv")


if __name__ == "__main__":
    main()
