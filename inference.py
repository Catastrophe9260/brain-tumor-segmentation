import argparse
import numpy as np
import torch
import onnxruntime as ort
from flask import Flask, request, jsonify

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
        torch.onnx.export(model, dummy_input, model_path, input_names=["input"], output_names=["output"], opset_version=13)

def create_flask_app():
    ort_session = ort.InferenceSession("brainseg.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json()
        arr = np.array(payload["image"], dtype=np.float32)
        if arr.ndim == 4: # no batch dimension
            arr = arr[None, ...]
        output = ort_session.run(None, {input_name: arr})[0]
        return jsonify({"logits": output.tolist()})

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="exports the model to ONNX format")
    parser.add_argument("--serve", action="store_true", help="runs the Flask app")

    args = parser.parse_args()
    if args.export:
        export_to_onnx(in_ch, out_ch, num_filters, num_heads, "final_weights.pt", "brainseg.onnx")
    if args.serve:
        app = create_flask_app()
        app.run(host="0.0.0.0", port=5000)