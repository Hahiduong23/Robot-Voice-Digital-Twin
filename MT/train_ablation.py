import os
import json
import time
import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import get_model
from utils import (
    count_parameters,
    estimate_flops,
    measure_latency,
    plot_learning_curves,
    plot_confusion_matrix,
    GradCAM,
    plot_gradcam_overlay
)
import matplotlib.pyplot as plt

def train_single_model(model_name, optimizer_type="AdamW", epochs=35, lr=0.001, device="cpu"):
    print(f"\n========================================================")
    print(f"Training [{model_name}] with [{optimizer_type}] (lr={lr}, epochs={epochs}, CosineAnnealing + LabelSmoothing)")
    print(f"========================================================")

    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=32, use_specaug=True)
    model = get_model(model_name, num_classes=len(meta["class_names"])).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_type == "SGD_Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_times": []
    }

    best_val_acc = -1.0
    best_weights = None
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (preds.argmax(dim=1) == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (preds.argmax(dim=1) == y_batch).sum().item()
                val_total += len(y_batch)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epoch_times"].append(epoch_time)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:2d}/{epochs}] (lr: {current_lr:.6f}) | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

    total_time = time.time() - total_start

    # Evaluate Best Weights on Test Set
    model.load_state_dict(best_weights)
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            test_loss += loss.item() * len(y_batch)
            predicted = preds.argmax(dim=1)
            test_correct += (predicted == y_batch).sum().item()
            test_total += len(y_batch)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())

    test_acc = (test_correct / test_total) * 100.0
    avg_test_loss = test_loss / test_total
    print(f">> Finished: Total Time: {total_time:.1f}s | Best Val Acc: {best_val_acc:.2f}% | Test Acc: {test_acc:.2f}% | Test Loss: {avg_test_loss:.4f}")

    params = count_parameters(model)
    flops = estimate_flops(model, device=device)
    latency = measure_latency(model, device="cpu")

    save_tag = f"{model_name}_{optimizer_type}"
    ckpt_path = f"MT/results/{save_tag}.pth"
    torch.save({
        "model_name": model_name,
        "optimizer_type": optimizer_type,
        "state_dict": best_weights,
        "history": history,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "params": params,
        "flops": flops,
        "latency_ms": latency
    }, ckpt_path)

    plot_learning_curves(history, f"MT/results/curves_{save_tag}.png", title=save_tag)

    return {
        "model_name": model_name,
        "optimizer_type": optimizer_type,
        "params": params,
        "flops_m": flops / 1e6,
        "latency_ms": latency,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": avg_test_loss,
        "total_train_time_s": total_time,
        "history": history,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "class_names": meta["class_names"]
    }

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("MT/results", exist_ok=True)

    # 1. Ablation Study: M1 -> M5 (35 epochs)
    ablation_models = [
        "M1_BaselineCNN",
        "M2_DropoutCNN",
        "M3_BatchNormCNN",
        "M4_VoiceResNet",
        "M5_SE_VoiceResNet"
    ]
    ablation_results = []
    print("\n=================== STARTING PART 1: ABLATION STUDY ===================")
    for m in ablation_models:
        res = train_single_model(m, optimizer_type="AdamW", epochs=35, device=device)
        ablation_results.append({
            "model_name": res["model_name"],
            "params": res["params"],
            "flops_m": res["flops_m"],
            "latency_ms": res["latency_ms"],
            "best_val_acc": res["best_val_acc"],
            "test_acc": res["test_acc"],
            "test_loss": res["test_loss"],
            "total_train_time_s": res["total_train_time_s"]
        })

    with open("MT/results/ablation_summary.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
    print("\n[Ablation Study Completed! Summary saved to MT/results/ablation_summary.json]")

    # 2. Optimizer Benchmark on Best Model (M5)
    print("\n=================== STARTING PART 2: OPTIMIZER BENCHMARK ===================")
    optimizers = ["SGD", "SGD_Momentum", "Adam", "AdamW"]
    optimizer_results = {}
    last_m5_res = None

    for opt in optimizers:
        res = train_single_model("M5_SE_VoiceResNet", optimizer_type=opt, epochs=35, device=device)
        optimizer_results[opt] = {
            "test_acc": res["test_acc"],
            "test_loss": res["test_loss"],
            "best_val_acc": res["best_val_acc"],
            "history": res["history"]
        }
        if opt == "AdamW":
            last_m5_res = res

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt, data in optimizer_results.items():
        plt.plot(data["history"]["train_loss"], label=f"{opt} (Test: {data['test_acc']:.1f}%)", linewidth=2)
    plt.title("Optimizer Comparison - Training Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Label Smoothing 0.1)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.subplot(1, 2, 2)
    for opt, data in optimizer_results.items():
        plt.plot(data["history"]["val_acc"], label=f"{opt} (Val: {data['best_val_acc']:.1f}%)", linewidth=2)
    plt.title("Optimizer Comparison - Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("MT/results/optimizer_comparison.png", dpi=200)
    plt.close()

    # 3. Confusion Matrix for Best Model (M5 + AdamW)
    print("\n=================== PART 3: CONFUSION MATRIX & XAI GRAD-CAM ===================")
    plot_confusion_matrix(
        last_m5_res["y_true"],
        last_m5_res["y_pred"],
        last_m5_res["class_names"],
        "MT/results/confusion_matrix_M5.png",
        title="Confusion Matrix - M5 SE_VoiceResNet (AdamW + CosineAnnealing)"
    )

    # 4. Grad-CAM for Mel-Spectrogram
    from dataset import get_dataloaders
    _, _, test_loader, _ = get_dataloaders()
    sample_x, sample_y = next(iter(test_loader))
    
    m5_model = get_model("M5_SE_VoiceResNet", num_classes=len(last_m5_res["class_names"]))
    ckpt = torch.load("MT/results/M5_SE_VoiceResNet_AdamW.pth", weights_only=False)
    m5_model.load_state_dict(ckpt["state_dict"])
    m5_model.eval()

    gradcam = GradCAM(m5_model, target_layer=m5_model.layer3.conv2)
    for idx in [0, 1]:
        single_x = sample_x[idx:idx+1]
        true_label = last_m5_res["class_names"][sample_y[idx].item()]
        cam, pred_class = gradcam.generate(single_x, target_class=sample_y[idx].item())
        pred_label = last_m5_res["class_names"][pred_class]
        plot_gradcam_overlay(
            single_x[0],
            cam,
            title=f"True: {true_label} | Pred: {pred_label}",
            save_path=f"MT/results/gradcam_sample_{idx}_{true_label}.png"
        )
    print("[SUCCESS] All upgrades successfully executed and visualized!")

if __name__ == "__main__":
    main()