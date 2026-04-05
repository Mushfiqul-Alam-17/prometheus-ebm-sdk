"""
Visualization engine for PROMETHEUS-EBM.

Provides exactly the V5 Notebook visual outputs:
- EDKI Scatter (Correctness vs Overconfidence Gap)
- Reliability Diagram (Brier Calibration breakdown)
- Epistemic Fingerprint Radar Chart
"""
import os
import math
import numpy as np
import pandas as pd

import matplotlib
# Headless safe backend for CI, Kaggle kernels, or Vercel edge functions
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_epistemic_radar(summary_df: pd.DataFrame, output_path: str = "epistemic_radar.png"):
    """
    [C11.5 Protocol Parity]
    Generates Epistemic Fingerprint Radar Chart comparing SDA, CA, RP, and Calibration Gap.
    """
    if len(summary_df) == 0:
        return None

    radar_df = summary_df.copy()
    if 'overconfidence_gap' not in radar_df.columns:
        radar_df['overconfidence_gap'] = 0.0

    radar_df['overconfidence_gap'] = pd.to_numeric(radar_df['overconfidence_gap'], errors='coerce')
    if radar_df['overconfidence_gap'].isna().all():
        radar_df['overconfidence_gap'] = 0.0

    metrics = ['sda', 'ca', 'rp', 'overconfidence_gap']
    labels = ['SDA', 'CA', 'RP', 'Calibration (1 - gap)']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in radar_df.iterrows():
        values = []
        for metric in metrics:
            val = pd.to_numeric(row.get(metric), errors='coerce')
            if metric == 'overconfidence_gap':
                # For calibration, we invert the gap so higher is better
                val = 1.0 - float(val) if pd.notna(val) else 1.0
                values.append(float(np.clip(val, 0, 1)))
            else:
                values.append(float(val) if pd.notna(val) else 0.0)
        values += values[:1]

        model_label = str(row.get('model', 'Unknown')).split('/')[-1]
        ax.plot(angles, values, linewidth=2, label=model_label)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_title('Epistemic Fingerprint Radar by Model\n(Higher is better on all axes)', pad=24)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def plot_reliability_diagram(item_level_df: pd.DataFrame, output_path: str = "reliability_diagram.png"):
    """
    [C11 Protocol Parity]
    Reliability diagram showing confidence vs observed accuracy bins.
    """
    if len(item_level_df) == 0 or 'confidence' not in item_level_df.columns:
        return None

    n_bins_cal = 10
    models = item_level_df['model'].unique()
    limit = min(len(models), 5)
    
    fig, axes = plt.subplots(1, limit, figsize=(4 * limit, 4), sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for ax, model_name in zip(axes, models[:limit]):
        mdf = item_level_df[item_level_df['model'] == model_name].copy()
        mdf['confidence'] = pd.to_numeric(mdf['confidence'], errors='coerce')
        mdf = mdf.dropna(subset=['confidence'])
        confs = mdf['confidence'].to_numpy(dtype=float)
        correct_series = pd.to_numeric(mdf['correctness_flag'], errors='coerce').fillna(0)
        correct = correct_series.to_numpy(dtype=float)
        
        short = str(model_name).split('/')[-1].split('@')[0][:15]
        
        if len(confs) == 0 or len(correct) == 0:
            ax.set_title(short, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            ax.legend(fontsize=7)
            continue

        bin_edges = np.linspace(0, 1, n_bins_cal + 1)
        bin_accs, bin_confs = [], []
        
        for k in range(n_bins_cal):
            mask = (confs >= bin_edges[k]) & (confs < bin_edges[k + 1])
            if k == n_bins_cal - 1:
                mask = mask | (confs == bin_edges[k + 1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(confs[mask].mean())

        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color='steelblue', label='Observed')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('Predicted Confidence')
        ax.set_title(short, fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Observed Accuracy')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    plt.suptitle('Reliability Diagram (Predicted Confidence vs Observed Accuracy)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def plot_edki_scatter(item_level_df: pd.DataFrame, output_path: str = "edki_scatter.png"):
    """
    [C11 Protocol Parity]
    EDKI Scatter Plot mapping Mean Correctness vs Overconfidence Gap.
    """
    if len(item_level_df) == 0:
        return None

    edki_df = item_level_df.groupby('model', as_index=False).agg(
        mean_correctness=('correctness_flag', lambda s: float(pd.to_numeric(s, errors='coerce').mean())),
        mean_confidence=('confidence', lambda s: float(pd.to_numeric(s, errors='coerce').mean()))
    )
    
    edki_df['overconfidence_gap'] = edki_df['mean_confidence'] - edki_df['mean_correctness']
    edki_df = edki_df.dropna(subset=['mean_correctness', 'overconfidence_gap']).reset_index(drop=True)

    if len(edki_df) == 0:
        return None

    plt.figure(figsize=(8, 6))
    plt.scatter(edki_df['mean_correctness'], edki_df['overconfidence_gap'], s=90, label='Models')
    
    for _, r in edki_df.iterrows():
        short_label = str(r['model']).split('/')[-1][:15]
        plt.text(float(r['mean_correctness']) + 0.002, float(r['overconfidence_gap']) + 0.002,
                 short_label, fontsize=8)

    if len(edki_df) >= 2 and edki_df['mean_correctness'].nunique() > 1:
        x = edki_df['mean_correctness'].to_numpy(dtype=float)
        y = edki_df['overconfidence_gap'].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        plt.plot(xs, m * xs + b, '--', linewidth=1.5, label='Trendline')

    plt.xlabel('Mean correctness')
    plt.ylabel('Overconfidence gap')
    plt.title('EDKI Scatter: Correctness vs Overconfidence Gap')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path
