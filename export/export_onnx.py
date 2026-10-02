from mmodel.denoiser import Denoiser, DenoiserProps
from mmodel.run_denoiser import make_inputs, randomize_output
from vmodel.tokenizer import Tokenizer, TokenizerProps
import os
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnxconverter_common import float16

OUT_DIR = "web/bench"

# Candidate widths for the browser speed test. Nothing is trained yet: the weights are
# random, because the time a frame takes depends on the shapes and not on the values.
VARIANTS = {
    "wide": (128, 256, 512),
    "mid": (128, 256, 384),
    "narrow": (96, 192, 384)
}

COND_INPUTS = ["tau", "tau_ctx", "actions", "level", "x_pos", "powerup", "state_mask"]


class Conditioning(nn.Module):
    def __init__(self, denoiser):
        super(Conditioning, self).__init__()
        self.denoiser = denoiser

    def forward(self, tau, tau_ctx, actions, level, x_pos, powerup, state_mask):
        return self.denoiser.conditioning(tau, tau_ctx, actions, level, x_pos, powerup, state_mask)


class Backbone(nn.Module):
    def __init__(self, denoiser):
        super(Backbone, self).__init__()
        self.denoiser = denoiser

    def forward(self, z_noisy, context, cond):
        return self.denoiser.denoise(z_noisy, context, cond)


def export(model, inputs, path):
    torch.onnx.export(
        model,
        kwargs=inputs,
        f=path,
        input_names=list(inputs.keys()),
        output_names=["output"],
        dynamo=True,
        external_data=False
    )


def run_onnx(path, inputs):
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {name: np.asarray(value) for name, value in inputs.items()}

    return session.run(None, feeds)[0]


def export_half(path, path_half):
    # The inputs and outputs stay float32 so the browser can fill them from a plain
    # Float32Array; only the weights and the maths in between drop to half precision.
    model_half = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True)
    onnx.save(model_half, path_half)


def main():
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, channels in VARIANTS.items():
        props = DenoiserProps({"channels": channels})
        denoiser = Denoiser(props).eval()
        randomize_output(denoiser)

        inputs = make_inputs(props, batch_size=1)
        cond_inputs = {key: inputs[key] for key in COND_INPUTS}

        path_cond = f"{OUT_DIR}/cond_{name}.onnx"
        path = f"{OUT_DIR}/denoiser_{name}.onnx"
        path_half = f"{OUT_DIR}/denoiser_{name}_fp16.onnx"

        with torch.no_grad():
            cond = denoiser.conditioning(**cond_inputs)
            expected = denoiser(**inputs).numpy()

        backbone_inputs = {"z_noisy": inputs["z_noisy"], "context": inputs["context"], "cond": cond}

        export(Conditioning(denoiser), cond_inputs, path_cond)
        export(Backbone(denoiser), backbone_inputs, path)
        export_half(path, path_half)

        # The check runs the two graphs chained the way the browser will, against the
        # single PyTorch forward, so a mismatch in how they are split shows up here.
        backbone_inputs["cond"] = run_onnx(path_cond, cond_inputs)

        error_cond = np.abs(backbone_inputs["cond"] - cond.numpy()).max()
        error = np.abs(run_onnx(path, backbone_inputs) - expected).max()
        error_half = np.abs(run_onnx(path_half, backbone_inputs) - expected).max()

        num_params = sum(param.numel() for param in denoiser.parameters())
        print(
            f"{name} {channels}  params {num_params / 1e6:.1f}M  "
            f"cond err {error_cond:.1e}  "
            f"fp32 {os.path.getsize(path) / 1e6:.0f}MB err {error:.1e}  "
            f"fp16 {os.path.getsize(path_half) / 1e6:.0f}MB err {error_half:.1e}  "
            f"output std {expected.std():.2f}"
        )

        assert error_cond < 1e-4
        assert error < 1e-4
        # Half precision carries about 3 decimal digits; 1% of the output scale is the
        # most that rounding through the whole network should cost.
        assert error_half < 0.01 * expected.std()

    tokenizer = Tokenizer(TokenizerProps()).eval()

    inputs = {"z": torch.randn(1, tokenizer.props.latent_channels, 28, 32)}
    path = f"{OUT_DIR}/decoder.onnx"

    export(tokenizer.decoder, inputs, path)

    with torch.no_grad():
        expected = tokenizer.decoder(**inputs).numpy()

    error = np.abs(run_onnx(path, inputs) - expected).max()
    print(f"decoder  fp32 {os.path.getsize(path) / 1e6:.1f}MB err {error:.1e}")

    assert error < 1e-4

    print("\nAll exports match PyTorch.")

if (__name__ == "__main__"):
    main()
