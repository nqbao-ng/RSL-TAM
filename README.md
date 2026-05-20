# DEDNet + RL-EMO final classifier

This modified version keeps the original DEDNet feature extraction, audio/visual encoders, and relational subgraph interaction modules, but replaces the final multimodal emotion classifier with an RL-EMO style Q-value classifier.

## Changed files

- `model.py`
  - Added `RLEMOClassifier`.
  - Added feature-level multimodal fusion before Q-value prediction.
  - Final prediction now uses `Q(s_t, a)` over emotion classes instead of the original sum of modality logits.
  - The original unimodal heads are kept only for the auxiliary fusion loss.

- `train.py`
  - Passes labels into the model so the RL reward can be computed.
  - Adds RL Bellman loss to the original fusion CE + auxiliary modality losses.
  - Adds arguments:
    - `--rl_gamma`: discount factor. Default: `0.95` for IEMOCAP/DailyDialog, `0.7` for MELD.
    - `--rl_mu`: Bellman target mixing coefficient. Default: `0.5`.
    - `--rl_loss_w`: RL loss weight. Default: `1.0`.

## Example commands

IEMOCAP:

```bash
python train.py --lr=0.0001 --batch-size=16 --hidden_dim=512 --windows=20 --epochs=100 --Dataset="IEMOCAP" --save_model_path="./IEMOCAP" --rl_gamma 0.95 --rl_mu 0.5 --rl_loss_w 1.0
```

MELD:

```bash
python train.py --lr=0.00005 --batch-size=8 --hidden_dim=256 --windows=5 --epochs=20 --Dataset="MELD" --save_model_path="./MELD" --rl_gamma 0.7 --rl_mu 0.5 --rl_loss_w 1.0
```
