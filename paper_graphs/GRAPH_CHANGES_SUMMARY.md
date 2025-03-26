# Graph Changes Summary

## Changes Made to Graph Generation Code

The following modifications were made to enhance scientific accuracy and remove potentially misleading annotations in the paper graphs:

### 1. Effect Size Comparison Graph
- Modified to use actual effect sizes rather than absolute values
- Added color coding based on effect direction (positive = green, negative = red)
- Added a zero line instead of effect size category lines
- Improved axis labels and title for scientific clarity
- Added symmetric y-axis limits for better interpretation
- Filtered out fractal dimension metric if present

### 2. Statistical Significance Graph
- Added annotations for adjusted p-values
- Improved labels for clearer interpretation
- Filtered out fractal dimension metric if present
- Renamed "Adjusted p-value" to "P-value adjusted for multiple tests" for clarity

### 3. Robustness Under Perturbations Graph
- Removed potentially misleading "phi advantage" annotation
- Maintained the underlying data and visualization to preserve scientific integrity

### 4. Entanglement Entropy Comparison Graph
- Removed highlighted "significant difference" regions
- Removed maximum difference annotation
- Changed x-axis label to include "arb. units" (arbitrary units) for scientific accuracy
- Maintained the core data visualization to preserve scientific integrity

## Scientific Integrity Improvements

These changes were made to ensure:

1. **Neutral Presentation**: Removed language that might imply phi has special properties without strong statistical evidence
2. **Scientific Accuracy**: Improved axis labels and annotations to better represent the data
3. **Data Integrity**: Maintained all underlying data while improving how it's presented
4. **Statistical Rigor**: Added more information about statistical adjustments for multiple testing

All graphs now present the data in a more scientifically neutral manner while maintaining the ability to visualize the relationships between different scaling factors.
