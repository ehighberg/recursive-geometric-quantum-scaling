#!/usr/bin/env python
"""
Generate a simple HTML report summarizing the findings.
"""

import os
import pandas as pd
from pathlib import Path
from constants import PHI

def generate_report():
    """Generate a simple HTML report summarizing the findings."""
    print("Generating HTML report...")
    
    # Create report directory
    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    
    # List of generated image files
    image_files = [
        "phi_significance_plots.png",
        "phi_significance_derivatives.png",
        "phi_significance_zoom.png",
        "fractal_topology_relation.png",
        "fractal_topology_phase_diagram.png",
        "fs_scaling_plots.png",
        "fs_scaling_combined.png",
        "summary_table.png",
        "phi_comparison_table.png",
        "entanglement_entropy_phi.png",
        "entanglement_entropy_unit.png",
        "entanglement_spectrum_phi.png",
        "entanglement_growth_phi.png",
        "wavepacket_evolution_phi.png",
        "wavepacket_evolution_unit.png",
        "wavepacket_spacetime_phi.png",
        "phi_vs_unit_comparison.png"
    ]
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recursive Geometric Quantum Scaling: Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333366; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
            .caption {{ font-style: italic; margin-top: 10px; }}
            .section {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>Recursive Geometric Quantum Scaling: Comprehensive Report</h1>
        
        <div class="section">
            <h2>1. Fractal Recursion Analysis</h2>
            <p>
                This section presents the analysis of fractal properties in the quantum system,
                with special focus on the golden ratio (φ = {PHI:.6f}) as a scaling factor.
            </p>
    """
    
    # Add figures to HTML content
    for image in image_files:
        if os.path.exists(image):
            # Copy file to report directory
            with open(image, "rb") as src_file:
                with open(report_dir / image, "wb") as dst_file:
                    dst_file.write(src_file.read())
            
            # Add to HTML
            caption = image.replace(".png", "").replace("_", " ").title()
            html_content += f"""
            <div class="figure">
                <img src="{image}" alt="{caption}">
                <div class="caption">{caption}</div>
            </div>
            """
    
    # Add summary tables if available
    if os.path.exists("summary_table.csv"):
        try:
            df = pd.read_csv("summary_table.csv")
            html_content += f"""
            <div class="section">
                <h2>Summary Table</h2>
                {df.to_html(index=False)}
            </div>
            """
        except Exception as e:
            print(f"Error reading summary table: {e}")
    
    if os.path.exists("phi_comparison_table.csv"):
        try:
            df = pd.read_csv("phi_comparison_table.csv")
            html_content += f"""
            <div class="section">
                <h2>Phi Comparison Table</h2>
                {df.to_html(index=False)}
            </div>
            """
        except Exception as e:
            print(f"Error reading phi comparison table: {e}")
    
    # Close HTML
    html_content += """
        <div class="section">
            <h2>Conclusion</h2>
            <p>
                This report demonstrates the interplay between fractal recursion, topological protection,
                and the variable scale factor f_s in the quantum system. The golden ratio (φ) appears to
                play a special role in the system's behavior, as evidenced by the analyses presented above.
            </p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(report_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report generated in {report_dir / 'report.html'}")

if __name__ == "__main__":
    generate_report()
