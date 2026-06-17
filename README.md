## Dataset

We use the processed IEMOCAP and MELD [dataset](https://drive.google.com/drive/folders/1bw18Fy5FGLGwp1cpTH2QRohSY-Ewc-Rf?usp=sharing) for training and evaluation.

## Training

IEMOCAP:

```bash
python train.py --lr=0.0001 --batch-size=16 --hidden_dim=512 --windows=20 --epochs=100 --Dataset="IEMOCAP" --save_model_path="./IEMOCAP" --rl_gamma 0.9 --rl_mu 0.5 --rl_loss_w 0.1 --n_head=8 
```
MELD:

```bash
python train.py --lr=0.00005 --batch-size=8 --hidden_dim=256 --windows=5 --epochs=20 --Dataset="MELD" --save_model_path="./MELD" --rl_gamma 0.8 --rl_mu 0.5 --rl_loss_w 1.0
```

## Inference

```bash
python inference.py --Dataset "IEMOCAP" --model_path "./IEMOCAP/checkpoint_01.pth"
```
