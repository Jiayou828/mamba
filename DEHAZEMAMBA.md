# DehazeMamba reproduction

This branch adds a paired optical/SAR implementation of DehazeMamba to the
Mamba source checkout. It follows the paper's DM-T stage depths `[2, 2, 2, 1,
1]`, HPDM and PFM data flow, AdamW learning rate `2e-4`, cosine schedule to
`1e-6`, batch size 6, 150 epochs, and spatial plus frequency L1 loss with
frequency weight `0.1`.

## Reproduction assumptions

The article does not give the DM-T channel widths, selective-scan state size,
exact VSS implementation, FFT normalization, or AdamW weight decay. This
implementation records these choices in the model and checkpoint instead of
claiming they were specified: widths start at 24 and double by stage; the Mamba
state size is 8; the axial VSS uses shared bidirectional row/column scans; the
FFT is orthonormal; AdamW uses PyTorch's default weight decay. Fusion is applied
at the two deepest encoder levels, consistent with Fig. 2. These assumptions
are the main limits on exact numerical reproduction.

## Environment and commands

```bash
conda env create -f environment.yml
conda activate mamba
python tools/build_selective_scan.py --allow-major-mismatch
python train_dehaze.py --data-root /path/to/MRSHaze --epochs 150 --batch-size 6
python evaluate_dehaze.py --data-root /path/to/MRSHaze \
  --checkpoint outputs/dehazemamba/last.pt
```

The dataset layout is `MRSHaze/{train,test}/{GT,hazy,SAR}/*.png`; each image
pair is matched by filename. Training writes a resumable `last.pt` checkpoint
after every epoch, and evaluation writes per-image PSNR/SSIM plus predictions.

The local host has PyTorch built against CUDA 11.8 and a CUDA 12.0 compiler.
The helper's mismatch flag skips PyTorch's compile-time version guard; the
resulting extension was checked on the RTX 3090 with forward and backward
passes. On other machines, use a CUDA toolkit matching the PyTorch wheel and
omit `--allow-major-mismatch`.
