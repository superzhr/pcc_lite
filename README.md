# pcc-lite (SeedFormerLite, pure PyTorch, Windows-friendly)

This is a minimal, training-ready point cloud completion baseline that runs on **Windows** without compiling CUDA ops.
It uses a lightweight SeedFormer-like architecture and pure PyTorch Chamfer Distance.

## 1) Environment (Windows / Anaconda PowerShell)
```powershell
conda create -n pccwin python=3.10 -y
conda activate pccwin

# Install PyTorch that matches your GPU (CUDA 11.8 example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other deps
pip install -r requirements.txt
```

## 2) (Optional) Generate a tiny toy dataset to test the pipeline
```powershell
python -m pcc_lite.data_tools.make_toy_pcn --root ./data/PCN --n_train 60 --n_val 10 --n_test 10
```

## 3) Train & Eval
```powershell
python train.py
python eval.py
```

- Config is in `config.yaml`.
- PCN format expected: `data/PCN/{train,val,test}/*.h5` each containing datasets `partial (N,3)` and `gt (M,3)`.
- For real PCN data, just place files in those folders and run the same commands.

## Notes
- This baseline uses `torch.cdist` (O(N^2)) — keep point counts moderate.
- If VRAM is low, reduce `batch_size` or `up_ratio` in `config.yaml`.
- For research extensions: try density-aware Chamfer weights, adaptive seed generation, and 2-stage training.
