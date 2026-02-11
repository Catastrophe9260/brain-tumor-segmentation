import argparse
import torch
import onnxruntime as ort

from model import ResAtt3DUNet
from hyperparameters import IN_CH, OUT_CH, NUM_FILTERS, NUM_HEADS

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# hyperparameters
in_ch = IN_CH
out_ch = OUT_CH
num_filters = NUM_FILTERS
num_heads = NUM_HEADS

def export_to_onnx(in_ch, out_ch, num_filters, num_heads, weights_path, model_path):
    model = ResAtt3DUNet(
        in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads, dropout=False
    )

    model.to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))

    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn((1, 3, 64, 64, 64)).to(device)
        torch.onnx.export(model, dummy_input, model_path, input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="exports the model to ONNX format")
    parser.add_argument("--predict", action="store_true", help="runs a prediction")

    args = parser.parse_args()
    if args.export:
        export_to_onnx(in_ch, out_ch, num_filters, num_heads, "best_weights.pt", "brainseg.onnx")