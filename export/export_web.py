from export.export_onnx import Conditioning, Backbone, export, export_half, run_onnx, COND_INPUTS
from mmodel.train_dynamics import load_denoiser, CHECKPOINT as DYNAMICS_CHECKPOINT
from vmodel.tokenizer import Tokenizer, TokenizerProps, colorize
from vmodel.train_tokenizer import CHECKPOINT as TOKENIZER_CHECKPOINT
from envs.smb import load
from envs.smb_sequences import LATENT_STATS, FRAME_SKIP
from simul.build_assets import OUT_DIR as ASSET_DIR
from simul.engine import Game, NUM_LEVELS
from simul.run_replica import Bot
import gzip
import json
import os
import zlib

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

OUT_DIR = "web/play"
TRACE_LEVELS = (0, 2, 5, 16)  # 1-1, 1-3 (platforms), 2-2 (water), 5-1 (enemies)
TRACE_STEPS = 400


class Encode(nn.Module):
    # rgb frame in [0, 1] -> normalized latent, the form the dynamics model works in
    def __init__(self, tokenizer, mean, std):
        super(Encode, self).__init__()
        self.encoder = tokenizer.encoder
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, rgb):
        mu, logvar = self.encoder(rgb)

        return (mu - self.mean) / self.std


class Decode(nn.Module):
    # normalized latent -> colour number per pixel. The argmax is inside the graph so
    # the browser reads back 57 thousand numbers per frame instead of 3.6 million logits.
    def __init__(self, tokenizer, mean, std):
        super(Decode, self).__init__()
        self.decoder = tokenizer.decoder
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, z):
        logits = self.decoder(z * self.std + self.mean)

        return logits.argmax(dim=1).float()


def export_models():
    tokenizer = Tokenizer(TokenizerProps())
    tokenizer.load_state_dict(torch.load(TOKENIZER_CHECKPOINT, map_location="cpu"))
    tokenizer.eval()

    denoiser = load_denoiser(DYNAMICS_CHECKPOINT, "cpu")
    props = denoiser.props

    stats = np.load(LATENT_STATS)
    mean = torch.from_numpy(stats[0]).float()[None, :, None, None]
    std = torch.from_numpy(stats[1]).float()[None, :, None, None]

    data = load()
    palette = torch.from_numpy(data["palette"]).float() / 255.0
    frame = torch.from_numpy(np.asarray(data["frames"][5000])).long()[None]
    rgb = colorize(frame, palette)

    encode = Encode(tokenizer, mean, std).eval()
    decode = Decode(tokenizer, mean, std).eval()

    with torch.no_grad():
        z = encode(rgb)
        decoded = decode(z)

    export(encode, {"rgb": rgb}, f"{OUT_DIR}/encoder.onnx")
    export(decode, {"z": z}, f"{OUT_DIR}/decoder.onnx")

    error = np.abs(run_onnx(f"{OUT_DIR}/encoder.onnx", {"rgb": rgb}) - z.numpy()).max()
    same = (run_onnx(f"{OUT_DIR}/decoder.onnx", {"z": z}) == decoded.numpy()).mean()
    exact = (decoded.long() == frame).float().mean()
    print(f"encoder err {error:.1e}  decoder {same:.2%} of pixels as PyTorch  round trip {exact:.2%} of the frame")

    assert error < 1e-3 and same > 0.999 and exact > 0.98

    L, C = props.context_frames, props.latent_channels
    torch.manual_seed(0)

    inputs = {
        "z_noisy": torch.randn(1, C, 28, 32),
        "context": torch.randn(1, L, C, 28, 32),
        "tau": torch.tensor([0.6]),
        "tau_ctx": torch.tensor([0.1]),
        "actions": torch.randint(0, 2, (1, L + 1, props.action_dim)).float(),
        "level": torch.tensor([3]),
        "x_pos": torch.tensor([0.2]),
        "powerup": torch.tensor([1]),
        "state_mask": torch.ones(1)
    }
    cond_inputs = {key: inputs[key] for key in COND_INPUTS}

    with torch.no_grad():
        cond = denoiser.conditioning(**cond_inputs)
        expected = denoiser(**inputs).numpy()

    backbone_inputs = {"z_noisy": inputs["z_noisy"], "context": inputs["context"], "cond": cond}

    export(Conditioning(denoiser), cond_inputs, f"{OUT_DIR}/cond.onnx")
    export(Backbone(denoiser), backbone_inputs, f"{OUT_DIR}/denoiser_fp32.onnx")
    export_half(f"{OUT_DIR}/denoiser_fp32.onnx", f"{OUT_DIR}/denoiser.onnx")

    # Chained the way the browser runs them: conditioning in float32, the UNet in half.
    backbone_inputs["cond"] = run_onnx(f"{OUT_DIR}/cond.onnx", cond_inputs)
    error = np.abs(run_onnx(f"{OUT_DIR}/denoiser.onnx", backbone_inputs) - expected).max()
    os.remove(f"{OUT_DIR}/denoiser_fp32.onnx")

    print(f"denoiser fp16 {os.path.getsize(f'{OUT_DIR}/denoiser.onnx') / 1e6:.0f}MB err {error:.1e}  output std {expected.std():.2f}")
    assert error < 0.02 * expected.std()

    return {"context_frames": L, "latent_channels": C, "action_dim": props.action_dim, "frame_skip": FRAME_SKIP}


def export_assets(model_info):
    # Everything the engine loads, as one gzipped block of bytes plus a JSON index of
    # where each array sits in it. All the arrays are uint8.
    blob = bytearray()

    def put(array):
        array = np.ascontiguousarray(array, dtype=np.uint8)
        entry = {"offset": len(blob), "shape": list(array.shape)}
        blob.extend(array.tobytes())

        return entry

    levels = []

    for level in range(NUM_LEVELS):
        data = np.load(f"{ASSET_DIR}/level_{level:02d}.npz")
        ends = np.cumsum(data["lift_lengths"])

        levels.append({
            "background": put(data["background"]),
            "tiles": put(data["tiles"]),
            "hud": put(data["hud"]),
            "pieces": data["pieces"].tolist(),
            "spawns": data["spawns"].tolist(),
            "lift_kinds": data["lift_kinds"].tolist(),
            "lift_paths": [data["lift_paths"][end - length:end].tolist() for end, length in zip(ends, data["lift_lengths"])],
            "backdrop": int(data["backdrop"]),
            "area_type": int(data["area_type"])
        })

    sprites = np.load(f"{ASSET_DIR}/sprites.npz")

    index = {
        "model": model_info,
        "levels": levels,
        "mario_keys": sprites["mario_keys"].tolist(),
        "mario": put(sprites["mario"]),
        "object_keys": sprites["object_keys"].tolist(),
        "objects": put(sprites["objects"]),
        "block_keys": sprites["block_keys"].tolist(),
        "blocks": put(sprites["blocks"]),
        "digits": put(sprites["digits"]),
        "palette": sprites["palette"].tolist()
    }

    with open(f"{OUT_DIR}/assets.bin", "wb") as f:
        f.write(gzip.compress(bytes(blob)))

    with open(f"{OUT_DIR}/assets.json", "w") as f:
        json.dump(index, f)

    print(f"assets: {len(blob) / 1e6:.1f}MB of arrays, {os.path.getsize(f'{OUT_DIR}/assets.bin') / 1e6:.2f}MB gzipped")


def export_trace():
    # What the Python engine does with a fixed run of buttons, for web/play/test_engine.mjs
    # to hold the JavaScript engine to: the checksum of every frame it draws.
    traces = []

    for level in TRACE_LEVELS:
        game = Game(level)
        bot = Bot()
        actions, checksums = [], []

        for _ in range(TRACE_STEPS):
            action = bot(game)
            frame = game.step(action)

            actions.append(action.tolist())
            checksums.append(zlib.crc32(frame.tobytes()))

        traces.append({"level": level, "actions": actions, "checksums": checksums})

    os.makedirs(f"{OUT_DIR}/test", exist_ok=True)

    with open(f"{OUT_DIR}/test/trace.json", "w") as f:
        json.dump(traces, f)

    print(f"trace: {TRACE_STEPS} steps of levels {TRACE_LEVELS}")


def main():
    onnxruntime.disable_telemetry_events()
    os.makedirs(OUT_DIR, exist_ok=True)

    model_info = export_models()
    export_assets(model_info)
    export_trace()

    print("\nAll exports match PyTorch.")

if (__name__ == "__main__"):
    main()
