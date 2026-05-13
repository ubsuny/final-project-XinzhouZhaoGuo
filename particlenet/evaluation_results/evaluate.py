#!/usr/bin/env python
"""
Evaluation script for ParticleNet, DeepAK8, and MLP models.
Reads the prediction .root files and produces ROC curves comparing the three models.

Usage:
    python evaluate.py                    # produce ROC plot (default)
    python evaluate.py --print-info       # print branch info and summary statistics
    python evaluate.py --print-events N   # print first N events from each file

Reference: https://cms-ml.github.io/documentation/inference/particlenet.html
           Section "3. Evaluation of models"
"""

import argparse
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Model definitions: name, file, color (matching the reference plot style)
MODELS = [
    ('MLP',         'mlp_predict.root',         'blue'),
    ('DeepAK8',     'deepak8_predict.root',     'red'),
    ('ParticleNet', 'particlenet_predict.root',  'green'),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(filename):
    """Load truth labels and prediction scores from a .root file."""
    filepath = os.path.join(SCRIPT_DIR, filename)
    tree = uproot.open(filepath)['Events']
    arrays = tree.arrays(library='np')
    return arrays


def print_info():
    """Print branch information and summary statistics for all files."""
    for name, filename, _ in MODELS:
        data = load_data(filename)
        n_entries = len(data['is_signal_new'])
        n_signal = np.sum(data['is_signal_new'])
        n_bkg = np.sum(data['is_bkg'])
        print(f'\n{"="*60}')
        print(f'{name} ({filename})')
        print(f'{"="*60}')
        print(f'  Entries: {n_entries}')
        print(f'  Signal:  {n_signal}  ({100*n_signal/n_entries:.1f}%)')
        print(f'  Bkg:     {n_bkg}  ({100*n_bkg/n_entries:.1f}%)')
        print(f'\n  {"Branch":<25s} {"dtype":>8s} {"min":>12s} {"max":>12s} {"mean":>12s}')
        print(f'  {"-"*25} {"-"*8} {"-"*12} {"-"*12} {"-"*12}')
        for branch, arr in data.items():
            print(f'  {branch:<25s} {str(arr.dtype):>8s} {np.min(arr):>12.4f} {np.max(arr):>12.4f} {np.mean(arr):>12.4f}')


def print_events(n_events):
    """Print the first N events from each file."""
    for name, filename, _ in MODELS:
        data = load_data(filename)
        print(f'\n{"="*60}')
        print(f'{name} ({filename}) — first {n_events} events')
        print(f'{"="*60}')
        branches = list(data.keys())
        header = '  '.join(f'{b:>20s}' for b in branches)
        print(f'  {header}')
        for i in range(min(n_events, len(data[branches[0]]))):
            row = '  '.join(f'{data[b][i]:>20.6f}' for b in branches)
            print(f'  {row}')


def make_roc_plot():
    """Produce the ROC curve plot comparing all three models."""
    fig, ax = plt.subplots(figsize=(8, 8))

    results = []  # store (name, auc, acc_at_half, 1/bkg_eff_at_half) for the summary table

    for name, filename, color in MODELS:
        data = load_data(filename)
        y_true = data['is_signal_new'].astype(int)
        y_score = data['score_is_signal_new']

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Signal efficiency = tpr, Background efficiency = fpr
        sig_eff = tpr
        bkg_eff = fpr

        # Accuracy at score > 0.5 threshold
        pred_label = (y_score > 0.5).astype(int)
        accuracy = np.mean(pred_label == y_true)

        # 1/bkg_eff at 30% signal efficiency (background rejection)
        idx_30 = np.argmin(np.abs(sig_eff - 0.3))
        bkg_rej_at_30 = 1.0 / bkg_eff[idx_30] if bkg_eff[idx_30] > 0 else float('inf')

        results.append((name, roc_auc, accuracy, bkg_rej_at_30))

        ax.plot(sig_eff, bkg_eff, color=color, linewidth=1.5,
                label=f'{name} (AUC: {roc_auc:.3f})')

    ax.set_yscale('log')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel(r'Signal efficiency $\varepsilon_S$', fontsize=14)
    ax.set_ylabel(r'Background efficiency $\varepsilon_B$', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    outpath = os.path.join(SCRIPT_DIR, 'roc.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nROC plot saved to: {outpath}')

    # Print summary table
    print(f'\n{"Model":<15s} {"AUC":>8s} {"Accuracy":>10s} {"1/ε_B@ε_S=0.3":>16s}')
    print(f'{"-"*15} {"-"*8} {"-"*10} {"-"*16}')
    for name, roc_auc, acc, rej in results:
        print(f'{name:<15s} {roc_auc:>8.3f} {acc:>10.3f} {rej:>16.0f}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare model predictions from .root files.')
    parser.add_argument('--print-info', action='store_true',
                        help='Print branch info and summary statistics')
    parser.add_argument('--print-events', type=int, default=0, metavar='N',
                        help='Print first N events from each file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip ROC plot generation')
    args = parser.parse_args()

    if args.print_info:
        print_info()

    if args.print_events > 0:
        print_events(args.print_events)

    if not args.no_plot:
        make_roc_plot()


if __name__ == '__main__':
    main()
