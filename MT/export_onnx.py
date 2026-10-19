# =========================================================================
# Export SOTA Midterm Model (M5_SE_VoiceResNet) to Standalone ONNX format for Web & Edge
# =========================================================================
import os
import sys
import torch
import onnx

sys.path.append('MT')
from models import get_model

def export_model(
    model_name='M5_SE_VoiceResNet',
    checkpoint_path='MT/results/M5_SE_VoiceResNet_AdamW.pth',
    output_onnx_path='MT/results/m5_voice_resnet.onnx'
):
    print(f'[INFO] Nạp mô hình {model_name} từ: {checkpoint_path}')
    model = get_model(model_name, num_classes=7)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 1, 128, 63)
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    print(f'[INFO] Đang xuất mô hình sang ONNX Standalone (Opset 17)...')
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_mel'],
        output_names=['logits'],
        dynamic_axes={
            'input_mel': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        dynamo=False
    )

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_kb = os.path.getsize(output_onnx_path) / 1024
    print(f'[SUCCESS] Đã tạo thành công mô hình ONNX tự thân ({model_name})!')
    print(f' - Đường dẫn: {output_onnx_path}')
    print(f' - Dung lượng file: {file_size_kb:.1f} KB ({file_size_kb/1024:.2f} MB)')

if __name__ == '__main__':
    export_model()
