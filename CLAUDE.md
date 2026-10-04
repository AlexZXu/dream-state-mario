# Playable Super Mario Bros world model

A neural network that *is* the game: it takes the last 8 frames plus the player's buttons and generates the next frame, with no emulator running. The end goal is a website where anyone can play it in their browser.

The approved plan (steps 0-7) is at `~/.claude/plans/plan-out-the-steps-nifty-donut.md`. This file records where things stand as of 2026-10-02 and what differs from that plan.

## Working agreements

- Stop at the end of each plan step. Alexander reviews the code and commits it himself; do not commit.
- Run your own checks before handing code over. The testing style is an assert-based `run_*.py` next to the code, ending in `print("\nAll tests passed.")`, not pytest.
- Code style: the `alex-code-style` skill (parenthesized conditions, `super(Class, self).__init__()`, keyword args on named layers, ALL_CAPS module constants, comments that explain why).
- Long jobs: launch detached (`nohup caffeinate -i .venv/bin/python -u -m ... > data/logs/x.log 2>&1 &`) so they survive a session stop.

## Decisions made

- **Game:** Super Mario Bros (NES), all 32 levels.
- **Data:** no ROM is available, so training uses the public dataset `rafaelcp/smbdataset` (CC-BY-4.0): 737,134 frames, 280 episodes, one player, 60 fps, with buttons and a 2 KB RAM snapshot per frame. The emulator route (`envs/mario.py`, `play_record.py`, `collect_rollouts.py`) is written but untested against Mario and unused.
- **Budget:** prototype on the M5 MacBook (24 GB, MPS), then about $100-300 of rented single-GPU time for the full model.
- **Inference:** in the visitor's browser with onnxruntime-web on WebGPU.
- **Hybrid design:** the model learns everything inside a level; spawn, respawn, lives and level changes are ordinary code driven by a small "reader" network.

## Architecture

Model step = 3 NES frames (20 Hz). Frames are 224x256, stored as NES colour numbers (64-colour palette).

- **V, tokenizer** (`vmodel/tokenizer.py`): conv autoencoder, frame -> 8x28x32 latent. Encoder 3.1M params, decoder 1.1M. The decoder outputs 64-way logits per pixel (cross-entropy), not RGB.
- **M, dynamics** (`mmodel/denoiser.py`, `mmodel/flow.py`): UNet denoiser trained with rectified flow. Input is the noisy next latent stacked with 8 context latents. Conditioning through adaptive GroupNorm: noise level, context noise level, buttons for the last 9 steps (18 values per step: 6 buttons x 3 recorded frames), and level id / x-position / power-up with 25% dropout. Context is noised during training against drift.
- **R, reader** (`rmodel/reader.py`): 0.38M-param CNN on the last two latents -> scroll per step, died, level clear, power-up.
- **Harness** (`play/harness.py`): lives, level, x-position, respawn / next level / game over, with debouncing.

## Status by plan step

| Step | State |
|---|---|
| 0 Setup, browser speed budget | Done, except timing on a weaker laptop |
| 1 Data | Done via the public dataset |
| 2 Tokenizer | Done, passes its gate |
| 3 Prototype dynamics | Trained and evaluated; **fails its gate in rollouts** (Mario decays after 20-40 steps); not yet played by hand |
| 4 Reader and harness | Partly done; death detection fails; not wired into the player |
| 5 Full model on a cloud GPU | Training script and `cloud/train.sh` ready; **run not started** (needs a rented machine) |
| 6 Few-step sampling, ONNX export | Not started (random-weight export exists) |
| 7 Website | Not started (only the timing page `web/bench/`) |

## Results so far

- **Browser timing** (headless Chrome, M5, fp16, random weights): 2 denoise steps + decode take 30.7 ms for the 46M model (budget 50 ms); 4 steps take 58 ms. So the site must run at 2 steps and distillation is probably required.
- **Tokenizer:** 99.77% of held-out pixels exact, 99.22% of non-background pixels; 99.13% with 20% latent noise. Trained 20k steps, 80 minutes.
- **Prototype dynamics:** 19.3M params (`CHANNELS = (64, 128, 256)`), all levels, 30k steps at batch 32, 2 h 44 min. Held-out flow loss 0.19, flat over the last 8k steps.
- **Prototype rollouts** (`eval.rollout_eval`, 16 held-out starts, real buttons): frames stay sharp for all 100 steps and scene changes work (pipe exit, flagpole), but Mario shrinks to a blob or vanishes after 20-40 steps, the scroll can stall, and newly scrolled-in scenery is invented (1-1 drifts into another world). Pixels exact: 93-97% at step 1 (tokenizer ceiling 99.8%), about 80% from step 20 on, no better than a frozen frame between steps 10 and 60. 4 / 8 / 16 Euler steps, context noise 0 / 0.1 / 0.3 and state conditioning off / frozen / true-from-RAM all land within noise of each other, so it is the model, not the sampler. `play.dream` runs at 18.6 fps on MPS with 4 steps.
- **Reader** (held-out): level clear 100% precision and recall; scroll off by 0.27 px per step; power-up 95.8%; **died 100% precision but 11% recall**.

## Next tasks, in order

1. **Evaluate the prototype:** `python -m eval.rollout_eval` (real buttons replayed through the model, compared with real frames over 100 steps, writes `data/samples/rollout.png`), then play it with `python -m play.dream`. Step 3's gate: run, jump, stomp and hit blocks for 2+ minutes without the image collapsing. Both scripts were only smoke-tested with random weights.
2. **Fix death detection.** The latent reader memorises training deaths. Max pooling, 4x more training and a 5x positive weight did not help. Untested idea: a reader on decoded pixels, where a sprite looks the same anywhere. Pit deaths are the hard case: the only sign is Mario absent for about 4 seconds. Castle levels (x-4) also have no "clear" label; they end at the axe with player state still "playing".
3. **Wire the harness into `play/dream.py`:** reader after each generated frame, `Harness.update`, re-seed from `level_starts` on respawn or next level, and feed the tracked level / x-position back as conditioning (`Dreamer.use_state`). `Dreamer.seed` currently only takes held-out dataset indices, so it needs a way to seed from an arbitrary frame. Then A/B state conditioning on vs off.
4. **Step 5:** train the full model (46M, all levels, current data) on a rented GPU. Decided on 2026-10-03 to do this before more data work, since the prototype was under-trained, not data-limited. On the machine: clone, copy `tokenizer.pth` over, `bash cloud/train.sh` (builds the dataset, encodes latents, launches `mmodel.train_dynamics` detached; re-running it resumes). `train_dynamics.py` now holds the full-run settings (300k steps, batch 64, bfloat16 autocast on CUDA; prototype values left commented) and refuses to start over a `dynamics.pth` from a different run. It writes EMA snapshots to `data/checkpoints/` every 25k steps; bring those and `dynamics.pth` back and compare them with `eval.rollout_eval`. The CUDA path (autocast, loader workers) has never run: only the MPS path was tested. On MPS the 46M model does about 0.5 steps per second, so the Mac is not an option.
5. **Step 6:** compare 2/3/4 Euler steps, distil to 2 if needed, export real weights. The export splits conditioning (fp32) from the UNet body (fp16) because the sinusoid embeddings break in half precision.
6. **Step 7:** the website.

## Things that are easy to get wrong

- **The data is thin and uneven:** 205 minutes total, 109 of them in world 8; 1-1 has 5 runs (about 3 minutes). Run is held 83% of the time, left 9%, down 0.4%. Expect the model to follow recorded routes well and respond poorly to unusual input.
- **Other public Mario datasets do not fit** (checked 2026-10-03): `maxmill/aq-mario-smb1` (4.9M frames of random and PPO play on 1-1) is a smoothed 224x224 resize at 2-frame skip, and undoing it recovers only about 80% of sprite pixels. The rest are tiny, speedruns at 15 fps, or button replays that need a ROM.
- **Train/validation split is by episode** (every 10th held out, `envs/smb_frames.split`), never by frame.
- **Tokenizer patches must be aligned to the 8-pixel grid.** Unaligned crops stalled it at 94%.
- **Tokenizer needs gradient clipping** (it diverged at step 1,050 without) and KL weight 1e-2.
- **Retraining the tokenizer invalidates `data/smb/latents.npy`.** Delete it and re-run `python -m vmodel.encode_dataset` (the script resumes a partial file, so it will not overwrite an existing one).
- **Training loops must repeat epochs** (`mmodel.train_dynamics.batches`); one pass is fewer steps than the schedule.
- **Dataset RAM quirk, and `read_ram` undoes it the wrong way (found 2026-10-03, not fixed):** the recorder wrote the RAM in text mode, which turns every byte **10** into 13, 10. `envs/smb.read_ram` collapses the pair to 13, so in `ram.npy` every true 10 is stored as 13 (no byte is ever 10). Visible effect: level page 10 reads as page 13, so `x_pos` is 768 too large between 2560 and 2815. The fix is `replace(b"\r\n", b"\n")` plus re-running `envs.convert_smb`; it changes the x-position conditioning the dynamics model trains on. `simul/build_assets.py` works around it locally (`true_scroll`, `true_page`).
- **onnxruntime telemetry** can abort Python at exit; `export_onnx.py` disables it.
- **WebGPU** only works on `localhost` or https.

## Backup: `simul/` (not a world model)

A stand-in to show if the real model is not ready: Super Mario Bros rebuilt as ordinary game code from the dataset, with a filter that makes it look generated. Nothing is learned or generated.

- `python -m simul.build_assets` (about 10 minutes, mostly disk reads) writes `data/replica/` (444 KB): per level a background stitched by per-pixel vote over the recorded frames, the solid-tile grid and enemy spawn points read from RAM, recorded platform paths; plus Mario, enemy and block drawings cut out of the frames.
- `simul/engine.py`: `Game.step(buttons) -> frame`, the same contract as `Dreamer.step`. Original physics constants, blocks, coins, mushroom, goombas and koopas (stomp only), platforms that replay their recorded path, flagpole, lives. Left out: fireballs, shells, piranha plants, hammer bros, firebars, Bowser, bonus rooms.
- A level only exists where some recording went. Where every player took a pipe (1-1, 2-1, 4-1, 5-2) the game cuts from the pipe to where it came out. 8-4 is its first room only.
- `simul/decay.py`: the world-model look, tuned by looking at frames from the real prototype (`mmodel.dreamer` run headless). Three parts: (1) an uncertainty map on the 28x32 latent grid, hot under moving sprites and at the newly scrolled-in edge, where pixels copy a neighbour or keep the last frame; (2) rot: cells with something in them start to go, small details dissolve into the backdrop and the inside of big structures melts into its main colour, then slowly comes back; cells next to Mario are kept whole; (3) HUD digits stay sharp but go wrong: the clock stalls and jumps, digits turn into lookalikes. Flat areas stay exact. About 1.3% of pixels differ on average.
- `python -m simul.play`: D toggles the decay (exact engine frame vs decayed), `[` `]` change its strength, N skips a level, R restarts. `python -m simul.run_replica` is the check.
- **Simulation + model together** (2026-10-03): `Dreamer.step(action, guide=frame, tau_guide, anchor)` starts generation from the simulation's frame noised to `tau_guide` instead of from pure noise (`mmodel.flow.sample` takes `guide`); `anchor` puts the simulation's frame, not the generated one, into the context. `Dreamer.seed_frames` seeds from any frame, which next task 3 also needed. `python -m simul.run_redream` compares settings on 200 bot steps of 5-1 and writes `data/samples/redream.png`. Pixels matching the simulation at steps 181-200, whole frame / Mario's box: model alone 80% / 70% (Mario lost, another level invented); true history only 94% / 74%; guide 0.8 94% / 86%; guide 0.6 99.7% / 94%; guide 0.4 99.9% / 97%. `python -m simul.redream` plays it (default guide 0.6, 19 fps on MPS with 4 steps; `[` `]` change the noise, H true history, D exact frame). The prototype erases a Mario who stands still (even at guide 0.4 only 45% of his pixels survive 30 steps), so `Redream` has two hard rules: after a respawn or level start the exact frame is shown until the first button, and while Mario moves slower than 1 px per frame the step uses true history and guide noise of at most 0.4 (99% of his pixels kept after stopping, about 72% while setting off). The HUD rows get their own, higher guide noise (`tau_hud`, default 0.7; `-` `=` in the player; `mmodel.flow.sample` takes a `loose` mask and spends one extra denoise step on it): at the frame's 0.6 the score and clock were always right, which no world model manages; at 0.7 they are mostly right, sometimes stall or show a neighbouring digit, and recover; from 0.8 the clock is rarely right. No retraining. What this is: a game engine with the network redrawing each frame, not a model that plays the game on its own.
- **In the browser** (2026-10-03): `web/play/` is `simul.redream` as a static page, with only the gameplay keys (arrows, X, Z). `python -m export.export_web` writes into it the prototype's real weights (`encoder.onnx`, `decoder.onnx` with the argmax inside, `cond.onnx` fp32, `denoiser.onnx` fp16; 60 MB, git-ignored), the levels and sprites (`assets.bin` gzipped + `assets.json`) and `test/trace.json`. `engine.js` is `simul/engine.py` line for line and `redream.js` is the guided sampling loop; a rule changed in one language must be changed in the other. Checks: `node web/play/test/test_engine.mjs` (1,600 frames over 4 levels identical to the Python engine) and `node web/play/test/test_browser.mjs` (headless Chrome, real models on WebGPU: 38 ms per frame on the M5, 99.3% of pixels matching the simulation). The page runs 2 Euler steps plus the HUD's extra one; 2, 3 and 4 steps give the same frames when starting from the guide, and 4 took 56 ms. To play: `python3 -m http.server 8000 --directory web/play`, then `localhost:8000`. Not deployed.
- A web port of the simulation alone needs no model: export `data/replica/` as PNG + JSON and rewrite engine and decay (about 850 lines) in JavaScript.

## Layout

- `envs/`: `smb.py` + `convert_smb.py` (dataset), `smb_frames.py` (tokenizer data), `smb_sequences.py` (dynamics windows), `smb_events.py` (reader labels), `mario.py` (RAM addresses, emulator wrapper)
- `vmodel/`, `mmodel/`, `rmodel/`: model, `train_*.py`, `run_*.py` checks. `mmodel/dreamer.py` is the generate-one-frame loop.
- `play/dream.py` (pygame player), `play/harness.py`, `eval/rollout_eval.py`, `export/export_onnx.py`, `export/export_web.py`, `web/bench/`, `web/play/`, `cloud/train.sh`
- `simul/`: the backup game above (`build_assets.py`, `engine.py`, `decay.py`, `play.py`, `run_replica.py`, `redream.py`, `run_redream.py`)
- Git-ignored: `tokenizer.pth`, `dynamics.pth`, `reader.pth` in the root; `data/smb/` (frames 42 GB, latents 10.6 GB, RAM 1.5 GB); `data/logs/`; `data/samples/`

Run everything from the project root with `.venv` (Python 3.12), as `python -m package.module`. Checks: `vmodel.run_tokenizer`, `mmodel.run_denoiser`, `mmodel.run_dynamics`, `rmodel.run_reader`, `envs.run_smb`, `envs.run_mario`, `simul.run_replica`.
