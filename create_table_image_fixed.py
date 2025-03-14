<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def create_table_image(df, title, output_path, header_color='#4CAF50', highlight_rows=None, highlight_color='#FFEB3B'):
    """
    Create an image from a DataFrame table with unbiased formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to convert to image.
    title : str
        Title of the table.
    output_path : Path or str
        Path to save the image.
    header_color : str
        Hex color for the header row.
    highlight_rows : list of int, optional
        Indices of rows to highlight (if any). Use for emphasizing specific data points regardless of content.
    highlight_color : str
        Hex color to use for highlighting rows.
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
    
    # Style highlighted rows if specified (supports multiple rows)
    if highlight_rows is not None:
        if isinstance(highlight_rows, int):
            highlight_rows = [highlight_rows]  # Convert single value to list
            
        for row_idx in highlight_rows:
            if 0 <= row_idx < len(df):
                for j in range(len(df.columns)):
                    cell = table[row_idx + 1, j]  # +1 for header row
                    cell.set_facecolor(highlight_color)
                    cell.set_text_props(fontweight='bold')
    
    # Style alternating rows for all non-highlighted rows
    for i in range(len(df)):
        # Skip if this row is highlighted
        if highlight_rows is not None and i in highlight_rows:
            continue
            
        # Apply alternating row coloring
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
=======
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
>>>>>>> 67621917d847af621febdd13bfc67b86a99b6e65
