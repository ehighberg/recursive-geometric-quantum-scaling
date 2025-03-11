#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def create_table_image(df, title, output_path, header_color='#4CAF50', highlight_row=None):
    """
    Create an image from a DataFrame table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert to image.
    title : str
        Title of the table.
    output_path : Path
        Path to save the image.
    header_color : str
        Hex color for the header row.
    highlight_row : int, optional
        Index of row to highlight (if any).
    """
    # Create figure and axis separately to avoid syntax issues
    fig = plt.figure(figsize=(12, len(df) * 0.6 + 1.5))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left',
        colLoc='left'
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style highlighted row if specified
    if highlight_row is not None and highlight_row < len(df):
        for j in range(len(df.columns)):
            cell = table[highlight_row + 1, j]  # +1 for header row
            cell.set_facecolor('#FFEB3B')
            cell.set_text_props(fontweight='bold')
    
    # Style alternating rows
    for i in range(len(df)):
        if i != highlight_row:  # Don't style if it's the highlight row
            for j in range(len(df.columns)):
                cell = table[i + 1, j]  # +1 for header row
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')
    
    # Add title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
