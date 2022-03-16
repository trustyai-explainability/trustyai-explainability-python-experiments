# experiments
Repository containing experiments for evaluation of TrustyAI Explainability Toolkit

## notebooks

Notebooks to reproduce available experiments:

* Original LIME implementation (discrete and continuous features setting) impact-score eval: https://colab.research.google.com/drive/1jLe-tdtE7uGQ0KIKMG4PWJjDgPPDn5W0#scrollTo=KjIRWtglSX0C

## scripts

### Train NAM model on FICO dataset

Install `nam` library from `https://github.com/AmrMKayid/nam`:

1. `git clone https://github.com/AmrMKayid/nam`
2. `cd nam`
3. `pip install .`

`cd` to `experiments` directory and run the training script

1. `python train_nam_pytorch_fico.py` 

Inside `saved_models/NAM_FICO` you will find NAM checkpoints that can be used for inference.
