import csv
from pathlib import Path
import matplotlib.pyplot as plt



def plot_losses(histories, title,out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (train_losses, val_losses) in histories.items():
        ax.plot(train_losses, linewidth=2, label=f"{label} train")
        ax.plot(val_losses, linewidth=2, label=f"{label} validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    history_dir = Path(out_path) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    file_name = (
        title.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    plot_path = history_dir / f"{file_name}.png"

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
def write_csv(rows,output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows: 
        print(f"Warning: no results to write to {output_path}")
        return
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
         writer = csv.DictWriter(f, fieldnames=rows[0].keys()) 
         writer.writeheader() 
         writer.writerows(rows)

