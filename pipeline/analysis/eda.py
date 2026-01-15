"""
Exploratory Data Analysis for ODIR dataset.

Analyzes the distribution of retinal disease labels in the dataset.

Author: Chidwipak Kuppani
"""

import pandas as pd


def analyze_dataset(csv_path='./data/preprocessed_filtered.csv'):
    """
    Perform basic exploratory analysis on the dataset.
    
    Args:
        csv_path: Path to the preprocessed CSV file
    """
    df = pd.read_csv(csv_path)
    
    print("Dataset Preview:")
    print(df.head())
    print("\n" + "=" * 50)
    
    print_label_distribution(df)
    print_label_combinations(df)


def print_label_distribution(df):
    """Print the count of each disease label."""
    print("\nLabel Distribution:")
    print("-" * 30)
    
    labels = {
        'A': 'Age-related Macular Degeneration',
        'C': 'Cataract',
        'D': 'Diabetic Retinopathy',
        'G': 'Glaucoma',
        'H': 'Hypertension',
        'M': 'Myopia',
        'O': 'Other Abnormalities',
        'N': 'Normal'
    }
    
    for code, name in labels.items():
        if code in df.columns:
            try:
                count = df[code].value_counts()[1]
                print(f'{name}: {count}')
            except KeyError:
                print(f'{name}: 0')


def print_label_combinations(df):
    """Print unique label combinations in the dataset."""
    print("\nLabel Combinations:")
    print("-" * 30)
    
    df = df.sort_values(by='ID')
    label_cols = [col for col in ['N', 'A', 'G', 'H', 'D', 'C', 'M'] if col in df.columns]
    grouped = df.groupby(label_cols).size().reset_index(name='Count')
    print(grouped)


if __name__ == "__main__":
    analyze_dataset()
