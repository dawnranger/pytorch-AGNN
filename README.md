# Attention-based Graph Neural Network in Pytorch

This repo attempts to reproduce the AGNN model described in [Attention-based Graph Neural Network for semi-supervised learning, under review at ICLR 2018](https://openreview.net/pdf?id=rJg4YGWRb)

## Premise
This code implements the exact model and experimental setup described in the paper, but I haven't been able to reproduce their exact results yet.

Ideally the model should reach a 82.6% accuracy on the Cora dataset, with the experimental setup described in the paper and implemented in the code (140 training nodes, 500 validation nodes, 1000 test nodes). If you manage to run the same setup of the paper, let me know your results.

## Requirements

* PyTorch 0.3.0
* Python 2.7

## Usage
```python train.py```

## Acknowledgment
This repo borrows plenty of code from [this repo by Thomas Kipf](https://github.com/tkipf/pygcn).
