# =========================================================================
# Train SOTA BC-ResNet (Qualcomm) with AdamW, Cosine Annealing, SpecAugment
# =========================================================================
import os
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append('MT')
from dataset import get_dataloaders
from models import get_model
from utils import count_parameters, estimate_flops, measure_latency, plot_learning_curves, plot_confusion_matrix

def train_bc_resnet(epochs=35, lr=0.001, device='cpu'):
    print('=========================================================================')
    print('       HUẤN LUYỆN SOTA BC-RESNET (QUALCOMM KEYWORD SPOTTING)')
    print('=========================================================================')

    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=32, use_specaug=True)
    model = get_model('BC_ResNet', num_classes=len(meta['class_names'])).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1, 1, 128, 63))
    latency = measure_latency(model, input_size=(1, 1, 128, 63), device=device)

    print(f'Mô hình: BC_ResNet | Tham số: {params:,} | FLOPs: {flops:.2f} M | Trễ CPU: {latency:.2f} ms')
    print('Bắt đầu huấn luyện...')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = -1.0
    best_weights = None
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
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
        epoch_train_acc = train_correct / train_total * 100.0

        # Validation
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
        epoch_val_acc = val_correct / val_total * 100.0
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_times'].append(epoch_time)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch [{epoch:02d}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.1f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.1f}% ({epoch_time:.2f}s)')

    total_time = time.time() - total_start
    print(f'Huấn luyện hoàn tất trong {total_time:.1f}s! Best Val Acc: {best_val_acc:.2f}%')

    # Load best weights for Test evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            test_loss += loss.item() * len(y_batch)
            test_correct += (preds.argmax(dim=1) == y_batch).sum().item()
            test_total += len(y_batch)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    final_test_acc = test_correct / test_total * 100.0
    final_test_loss = test_loss / test_total

    print('=========================================================================')
    print(f'KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (140 MẪU):')
    print(f' - Test Accuracy: {final_test_acc:.2f}%')
    print(f' - Test Loss:     {final_test_loss:.4f}')
    print(f' - CPU Latency:   {latency:.2f} ms')
    print('=========================================================================')

    # Save Checkpoint
    out_dir = 'MT/results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'BC_ResNet_AdamW.pth')

    checkpoint = {
        'model_name': 'BC_ResNet',
        'optimizer_type': 'AdamW',
        'state_dict': best_weights,
        'history': history,
        'test_acc': final_test_acc,
        'test_loss': final_test_loss,
        'best_val_acc': best_val_acc,
        'params': params,
        'flops': flops,
        'latency_ms': latency,
        'total_train_time_s': total_time
    }
    torch.save(checkpoint, save_path)
    print(f'[SUCCESS] Đã lưu checkpoint tại: {save_path}')

    # Plot Curves & Confusion Matrix
    curves_path = os.path.join(out_dir, 'curves_BC_ResNet_AdamW.png')
    plot_learning_curves(history, 'BC_ResNet', 'AdamW', curves_path)

    cm_path = os.path.join(out_dir, 'confusion_matrix_BC_ResNet.png')
    plot_confusion_matrix(all_labels, all_preds, meta['class_names'], cm_path)

    return checkpoint

if __name__ == '__main__':
    train_bc_resnet()
