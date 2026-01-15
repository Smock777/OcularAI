"""
Descriptive Statistics and Visualization for ODIR dataset.

Generates visualizations for class distribution, label correlation,
and other dataset statistics.

Author: Chidwipak Kuppani
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Color scheme (teal and purple theme)
PRIMARY_COLOR = '#00897B'   # Teal
SECONDARY_COLOR = '#7B1FA2'  # Purple
ACCENT_COLOR = '#00ACC1'    # Cyan


def plot_class_distribution(df, save_path='class_distribution.png'):
    """
    Plot the distribution of each class in the dataset.
    
    Args:
        df: DataFrame with class columns
        save_path: Path to save the figure
    """
    class_counts = df[['N', 'D', 'C', 'M']].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.index, class_counts.values, 
                   color=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#E91E63'])
    
    plt.title('Class Distribution in ODIR Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for bar, val in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(val), ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_label_correlation(df, save_path='label_correlation.png'):
    """
    Plot heatmap of label correlations.
    
    Args:
        df: DataFrame with label columns
        save_path: Path to save the figure
    """
    label_corr = df[['N', 'D', 'C', 'M']].corr()
    
    print("Label Correlation Matrix:")
    print(label_corr)
    print()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(label_corr, annot=True, cmap='RdPu', fmt=".2f",
                linewidths=0.5, center=0)
    plt.title('Label Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def calculate_multilabel_statistics(df):
    """
    Calculate and print multi-label classification statistics.
    
    Args:
        df: DataFrame with label columns
    """
    labels = df[['N', 'D', 'C', 'M']]
    
    # Label cardinality: average number of labels per instance
    label_cardinality = labels.sum(axis=1).mean()
    print(f"Label Cardinality: {label_cardinality:.3f}")
    
    # Label density: label cardinality normalized by total labels
    label_density = label_cardinality / len(labels.columns)
    print(f"Label Density: {label_density:.3f}")
    
    # Label frequency: proportion of instances with each label
    label_frequency = labels.mean()
    print("\nLabel Frequency:")
    for label, freq in label_frequency.items():
        print(f"  {label}: {freq:.3f}")


def plot_label_cardinality(df, save_path='label_cardinality.png'):
    """
    Plot distribution of number of labels per instance.
    
    Args:
        df: DataFrame with label columns
        save_path: Path to save the figure
    """
    label_cardinality = df[['N', 'D', 'C', 'M']].sum(axis=1)
    
    plt.figure(figsize=(8, 6))
    counts = label_cardinality.value_counts().sort_index()
    bars = plt.bar(counts.index, counts.values, color=SECONDARY_COLOR)
    
    plt.title('Label Cardinality Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Labels per Instance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(counts.index)
    
    # Add value labels
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    """Run all statistical analyses and generate visualizations."""
    # Load dataset
    df = pd.read_csv('./data/final_dataset.csv')
    
    print("=" * 50)
    print("ODIR Dataset Statistics")
    print("=" * 50)
    print()
    
    # Generate all visualizations
    plot_class_distribution(df)
    plot_label_correlation(df)
    calculate_multilabel_statistics(df)
    plot_label_cardinality(df)


if __name__ == "__main__":
    main()