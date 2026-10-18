# =========================================================================
# FINAL TERM: Train SOTA BC-ResNet (Qualcomm AI Research) on Data_FN (2,800 samples)
# Optimizer: AdamW (lr=0.001, wd=1e-4) | Scheduler: Cosine Annealing (35 epochs)
# Augmentation: Google Brain SpecAugment (p=0.8) | Loss: Label Smoothing (0.1)
# =========================================================================
import os
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add root and MT to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'MT'))

from MT.dataset import get_dataloaders
from FN.models import get_model
from MT.utils import count_parameters, estimate_flops, measure_latency

def train_bc_resnet(epochs=35, lr=0.001, device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
    print('=========================================================================')
    print('   GIAI ĐOẠN CUỐI KỲ: HUẤN LUYỆN SOTA BC-RESNET (QUALCOMM KWS)')
    print('   TẬP DỮ LIỆU ĐỒNG BỘ: Data_FN (2,800 MẪU - 400 MẪU/LỚP)')
    print(f'   THIẾT BỊ HUẤN LUYỆN: {device.upper()}')
    print('=========================================================================')

    dataset_path = 'FN/dataset_processed_2800.pt'
    train_loader, val_loader, test_loader, meta = get_dataloaders(dataset_path=dataset_path, batch_size=32, use_specaug=True)
    model = get_model('BC_ResNet', num_classes=len(meta['class_names']))

    params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1, 1, 128, 63), device='cpu')
    latency = measure_latency(model, input_size=(1, 1, 128, 63), device='cpu')
    
    # Move model to target device for training
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f'Mô hình: BC_ResNet | Tham số: {params:,} | FLOPs: {flops:.2f} M | Trễ CPU: {latency:.2f} ms')
    print(f'Số mẫu: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}')
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
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (out.argmax(dim=1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100.0

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (out.argmax(dim=1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} - Acc: {train_acc:6.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc:6.2f}% | Best Val: {best_val_acc:6.2f}% | ({epoch_time:.2f}s)')

    total_time = time.time() - total_start
    print(f'\nHuấn luyện hoàn tất sau {total_time:.1f}s! Đang đánh giá trên tập Kiểm thử độc lập (Test Set)...')

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            test_correct += (out.argmax(dim=1) == y_batch).sum().item()
            test_total += X_batch.size(0)

    test_loss /= test_total
    test_acc = (test_correct / test_total) * 100.0

    print('=========================================================================')
    print(f' KẾT QUẢ TEST SET BC-RESNET (QUALCOMM KWS) TRÊN Data_FN:')
    print(f' - Best Val Accuracy:  {best_val_acc:.2f}% (trên 280 mẫu)')
    print(f' - Test Accuracy:      {test_acc:.2f}% (trên 280 mẫu kiểm thử)')
    print(f' - Test Loss:          {test_loss:.4f}')
    print(f' - Inference Latency:  {latency:.2f} ms')
    print('=========================================================================')

    # Save to FN/results
    os.makedirs('FN/results', exist_ok=True)
    out_pth = 'FN/results/BC_ResNet_AdamW.pth'
    torch.save({
        'model_name': 'BC_ResNet',
        'optimizer_type': 'AdamW',
        'state_dict': best_weights,
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_val_acc': best_val_acc,
        'params': params,
        'flops': flops,
        'latency_ms': latency,
        'total_train_time_s': total_time,
        'num_samples': 2800
    }, out_pth)
    print(f'[SUCCESS] Đã lưu checkpoint tại: {out_pth}')

    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    axes[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#dc2626', lw=2)
    axes[0].set_title('Đường cong Hàm Mất Mát (BC-ResNet AdamW - 2,800 Mẫu)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    axes[1].plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', color='#2563eb', lw=2)
    axes[1].plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', color='#16a34a', lw=2)
    axes[1].set_title(f'Độ chính xác (Best Val: {best_val_acc:.1f}% | Test: {test_acc:.1f}%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    curve_path = 'FN/results/curves_BC_ResNet_AdamW.png'
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f'[SUCCESS] Đã lưu biểu đồ học tập tại: {curve_path}')

if __name__ == '__main__':
    train_bc_resnet()
