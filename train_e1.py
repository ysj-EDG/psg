#!/usr/bin/env python3
"""Stable public entry for the recovered E1 training implementation.

The implementation is kept in train_domain_robust_encoder_full.py, whose source
is the historical full-training implementation from Matcha commit 7f2eae2.
It trains the EMA teacher/student encoder and writes checkpoints locally; no
checkpoint is part of this repository.
"""
from train_domain_robust_encoder_full import main


if __name__ == "__main__":
    main()
