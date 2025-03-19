#!/usr/bin/env python
"""
Scientific validator for statistical analysis of RGQS phenomena.

This module provides a rigorous framework for validating scientific hypotheses
related to phi-scaling effects using well-established statistical methods.
"""

import numpy as np
import scipy.stats as stats
from constants import PHI

class ScientificValidator:
    """
    A validator for scientific testing of phi-related phenomena.

    This class implements rigorous statistical methods for testing
    hypotheses related to phi-scaling effects addressing Tier 2 issues
    in the RGQS framework.
    """

    def __init__(self, alpha=0.05):
        """
        Initialize the scientific validator.

        Parameters
        ----------
        alpha : float, optional
            Significance level for hypothesis testing, by default 0.05
        """
        self.alpha = alpha

    def validate_multiple_metrics(self, metrics_data, correction_method='bonferroni'):
        """
        Validate multiple metrics with appropriate statistical testing.

        Parameters
        ----------
        metrics_data : dict
            Dictionary mapping metric names to data dictionaries.
            Each data dictionary maps scale factors to arrays of values.
        correction_method : str, optional
            Method for multiple testing correction:
            - 'bonferroni': Bonferroni correction (most conservative)
            - 'holm': Holm-Bonferroni step-down procedure (more powerful)
            - 'benjamini-hochberg': Controls false discovery rate
            By default 'bonferroni'.

        Returns
        -------
        dict
            Dictionary containing validation results.
        """
        results = {
            'individual_results': {},
            'combined_results': {
                'significant_metrics': [],
                'adjusted_p_values': {}
            }
        }

        # Process each metric
        for metric_name, metric_data in metrics_data.items():
            # Extract phi and control data
            phi_data, control_data = self._extract_data(metric_data)

            # Calculate statistics
            t_stat, p_value = self._calculate_p_value(phi_data, control_data)
            effect_size = self._calculate_effect_size(phi_data, control_data)
            effect_category = self._categorize_effect_size(effect_size)
            confidence_interval = self._calculate_confidence_interval(effect_size, phi_data, control_data)

            # Store results for this metric
            results['individual_results'][metric_name] = {
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_size_category': effect_category,
                'confidence_interval': confidence_interval,
                'is_significant': p_value < self.alpha,
                'significant_after_correction': p_value < self.alpha,  # Will be updated later
                'adjusted_p_value': p_value  # Will be updated later
            }

        # Apply multiple testing correction
        if len(results['individual_results']) > 1:
            adjusted_p_values = self._apply_multiple_testing_correction(
                {name: result['p_value'] for name, result in results['individual_results'].items()},
                correction_method
            )

            # Update results with corrected p-values
            for metric_name, adjusted_p_value in adjusted_p_values.items():
                results['individual_results'][metric_name]['adjusted_p_value'] = adjusted_p_value
                results['individual_results'][metric_name]['significant_after_correction'] = adjusted_p_value < self.alpha

                if adjusted_p_value < self.alpha:
                    results['combined_results']['significant_metrics'].append(metric_name)

                results['combined_results']['adjusted_p_values'][metric_name] = adjusted_p_value

        return results

    def _extract_data(self, metric_data):
        """
        Extract phi and control data from metric data.

        Parameters
        ----------
        metric_data : dict
            Dictionary mapping scale factors to arrays of values.

        Returns
        -------
        tuple
            Tuple containing phi data and control data arrays.
        """
        phi_data = None
        control_data = []

        for factor, values in metric_data.items():
            if factor == PHI or (isinstance(factor, str) and factor.startswith('np.float64')):
                phi_data = np.array(values)
            else:
                control_data.extend(values)

        control_data = np.array(control_data)

        return phi_data, control_data

    def _calculate_p_value(self, group1, group2):
        """
        Calculate p-value using Welch's t-test.

        Parameters
        ----------
        group1 : array-like
            First group of data.
        group2 : array-like
            Second group of data.

        Returns
        -------
        tuple
            Tuple containing t-statistic and p-value.
        """
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_value

    def _calculate_effect_size(self, group1, group2):
        """
        Calculate Cohen's d effect size.

        Parameters
        ----------
        group1 : array-like
            First group of data.
        group2 : array-like
            Second group of data.

        Returns
        -------
        float
            Cohen's d effect size.
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Calculate pooled standard deviation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)

        # Calculate Cohen's d
        d = (mean1 - mean2) / pooled_std

        return d

    def _categorize_effect_size(self, effect_size):
        """
        Categorize effect size according to Cohen's guidelines.

        Parameters
        ----------
        effect_size : float
            Cohen's d effect size.

        Returns
        -------
        str
            Effect size category.
        """
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'

    def _calculate_confidence_interval(self, effect_size, group1, group2, confidence=0.95):
        """
        Calculate confidence interval for effect size.

        Parameters
        ----------
        effect_size : float
            Cohen's d effect size.
        group1 : array-like
            First group of data.
        group2 : array-like
            Second group of data.
        confidence : float, optional
            Confidence level, by default 0.95

        Returns
        -------
        tuple
            Tuple containing lower and upper bounds of the confidence interval.
        """
        n1, n2 = len(group1), len(group2)

        # Calculate standard error of d
        se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))

        # Calculate critical value from t-distribution
        df = n1 + n2 - 2
        critical_value = stats.t.ppf((1 + confidence) / 2, df)

        # Calculate confidence interval
        lower = effect_size - critical_value * se
        upper = effect_size + critical_value * se

        return (lower, upper)

    def _apply_multiple_testing_correction(self, p_values, method='bonferroni'):
        """
        Apply multiple testing correction to p-values.

        Parameters
        ----------
        p_values : dict
            Dictionary mapping metric names to p-values.
        method : str, optional
            Method for multiple testing correction, by default 'bonferroni'.
            Options:
            - 'bonferroni': Bonferroni correction (most conservative)
            - 'holm': Holm-Bonferroni step-down procedure (more powerful)
            - 'benjamini-hochberg': Controls false discovery rate

        Returns
        -------
        dict
            Dictionary mapping metric names to adjusted p-values.
        """
        if method == 'bonferroni':
            # Bonferroni correction
            n = len(p_values)
            return {name: min(p_value * n, 1.0) for name, p_value in p_values.items()}

        elif method == 'holm':
            # Holm-Bonferroni step-down procedure
            n = len(p_values)

            # Sort p-values in ascending order
            sorted_p = sorted([(name, p_value) for name, p_value in p_values.items()],
                             key=lambda x: x[1])

            # Apply Holm's correction
            adjusted_p = {}
            for i, (name, p_value) in enumerate(sorted_p):
                adjusted_p[name] = min(p_value * (n - i), 1.0)

            # Ensure monotonicity
            for i in range(len(sorted_p) - 1, 0, -1):
                name1 = sorted_p[i][0]
                name2 = sorted_p[i-1][0]
                adjusted_p[name2] = min(adjusted_p[name2], adjusted_p[name1])

            return adjusted_p

        elif method == 'benjamini-hochberg':
            # Benjamini-Hochberg procedure
            n = len(p_values)

            # Sort p-values in ascending order
            sorted_p = sorted([(name, p_value) for name, p_value in p_values.items()],
                             key=lambda x: x[1])

            # Apply Benjamini-Hochberg procedure
            adjusted_p = {}
            for i, (name, p_value) in enumerate(sorted_p):
                adjusted_p[name] = p_value * n / (i + 1)

            # Ensure monotonicity
            for i in range(len(sorted_p) - 1, 0, -1):
                name1 = sorted_p[i][0]
                name2 = sorted_p[i-1][0]
                adjusted_p[name2] = min(adjusted_p[name2], adjusted_p[name1])

            # Cap at 1.0
            adjusted_p = {name: min(p_value, 1.0) for name, p_value in adjusted_p.items()}

            return adjusted_p

        else:
            # No correction (not recommended)
            print(f"Warning: Unknown correction method '{method}'. No correction applied.")
            return p_values

    def validate_single_metric(self, phi_data, control_data):
        """
        Validate a single metric with appropriate statistical testing.

        Parameters
        ----------
        phi_data : array-like
            Data for phi scaling.
        control_data : array-like
            Data for control conditions.

        Returns
        -------
        dict
            Dictionary containing validation results.
        """
        # Convert inputs to numpy arrays
        phi_data = np.array(phi_data)
        control_data = np.array(control_data)

        # Calculate statistics
        t_stat, p_value = self._calculate_p_value(phi_data, control_data)
        effect_size = self._calculate_effect_size(phi_data, control_data)
        effect_category = self._categorize_effect_size(effect_size)
        confidence_interval = self._calculate_confidence_interval(effect_size, phi_data, control_data)

        # Return results
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_category': effect_category,
            'confidence_interval': confidence_interval,
            'is_significant': p_value < self.alpha,
            't_statistic': t_stat,
            'phi_mean': np.mean(phi_data),
            'control_mean': np.mean(control_data),
            'phi_std': np.std(phi_data, ddof=1),
            'control_std': np.std(control_data, ddof=1),
            'phi_n': len(phi_data),
            'control_n': len(control_data)
        }
