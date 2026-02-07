"""
Statistical Significance Testing for A/B Tests
Implements various statistical tests to determine if differences are significant
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    interpretation: str = ""


class StatisticalAnalyzer:
    """
    Performs statistical analysis on A/B test results
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha

    def compare_confidence_scores(
        self,
        variant_a_scores: List[float],
        variant_b_scores: List[float],
        test_type: str = "ttest"
    ) -> StatisticalTestResult:
        """
        Compare confidence scores between two variants
        
        Args:
            variant_a_scores: List of confidence scores for variant A
            variant_b_scores: List of confidence scores for variant B
            test_type: Type of test ("ttest" or "mannwhitney")
            
        Returns:
            StatisticalTestResult
        """
        if test_type == "ttest":
            # Independent samples t-test
            statistic, p_value = stats.ttest_ind(
                variant_a_scores,
                variant_b_scores,
                equal_var=False  # Welch's t-test
            )
            test_name = "Welch's t-test"
        elif test_type == "mannwhitney":
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                variant_a_scores,
                variant_b_scores,
                alternative='two-sided'
            )
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        is_significant = p_value < self.alpha

        # Calculate effect size (Cohen's d)
        effect_size = self._cohens_d(variant_a_scores, variant_b_scores)

        # Interpretation
        mean_a = np.mean(variant_a_scores)
        mean_b = np.mean(variant_b_scores)

        if is_significant:
            if mean_a > mean_b:
                interpretation = f"Variant A has significantly higher confidence (p={p_value:.4f})"
            else:
                interpretation = f"Variant B has significantly higher confidence (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in confidence (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def compare_inference_times(
        self,
        variant_a_times: List[float],
        variant_b_times: List[float]
    ) -> StatisticalTestResult:
        """
        Compare inference times between two variants
        
        Args:
            variant_a_times: List of inference times for variant A (ms)
            variant_b_times: List of inference times for variant B (ms)
            
        Returns:
            StatisticalTestResult
        """
        # Use Mann-Whitney U test (times are often not normally distributed)
        statistic, p_value = stats.mannwhitneyu(
            variant_a_times,
            variant_b_times,
            alternative='two-sided'
        )

        is_significant = p_value < self.alpha

        median_a = np.median(variant_a_times)
        median_b = np.median(variant_b_times)

        if is_significant:
            if median_a < median_b:
                interpretation = f"Variant A is significantly faster (p={p_value:.4f})"
            else:
                interpretation = f"Variant B is significantly faster (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in speed (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Mann-Whitney U test (inference time)",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )

    def compare_sentiment_distributions(
        self,
        variant_a_counts: Dict[str, int],
        variant_b_counts: Dict[str, int]
    ) -> StatisticalTestResult:
        """
        Compare sentiment distributions using chi-square test
        
        Args:
            variant_a_counts: {"Positive": count, "Negative": count}
            variant_b_counts: {"Positive": count, "Negative": count}
            
        Returns:
            StatisticalTestResult
        """
        # Create contingency table
        observed = np.array([
            [variant_a_counts.get("Positive", 0), variant_a_counts.get("Negative", 0)],
            [variant_b_counts.get("Positive", 0), variant_b_counts.get("Negative", 0)]
        ])

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        is_significant = p_value < self.alpha

        # Calculate proportions
        total_a = sum(variant_a_counts.values())
        total_b = sum(variant_b_counts.values())

        pos_rate_a = variant_a_counts.get("Positive", 0) / total_a if total_a > 0 else 0
        pos_rate_b = variant_b_counts.get("Positive", 0) / total_b if total_b > 0 else 0

        if is_significant:
            interpretation = f"Significant difference in sentiment distribution (p={p_value:.4f}). "
            interpretation += f"A: {pos_rate_a:.1%} positive, B: {pos_rate_b:.1%} positive"
        else:
            interpretation = f"No significant difference in sentiment distribution (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Chi-square test (sentiment distribution)",
            statistic=chi2,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )

    def calculate_confidence_intervals(
        self,
        scores: List[float],
        confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean
        
        Args:
            scores: List of scores
            confidence: Confidence level (default: use self.confidence_level)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence is None:
            confidence = self.confidence_level

        mean = float(np.mean(scores))
        sem = float(stats.sem(scores))  # Standard error of the mean

        margin = float(sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1))

        return (mean - margin, mean + margin)

    def calculate_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_mean: Mean of baseline metric
            baseline_std: Standard deviation of baseline
            minimum_detectable_effect: Minimum effect size to detect (as fraction)
            power: Statistical power (default 0.8)
            
        Returns:
            Required sample size per variant
        """
        effect_size = minimum_detectable_effect * baseline_mean / baseline_std

        # Use statsmodels if available, otherwise approximate
        try:
            from statsmodels.stats.power import tt_ind_solve_power
            n = tt_ind_solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=power,
                alternative='two-sided'
            )
            return int(np.ceil(float(n)))
        except ImportError:
            # Approximate formula
            z_alpha = float(stats.norm.ppf(1 - self.alpha / 2))
            z_beta = float(stats.norm.ppf(power))
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(float(n)))

    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            group1: First group of scores
            group2: Second group of scores
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))

        # Pooled standard deviation
        pooled_std = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))

        # Cohen's d
        d = float((np.mean(group1) - np.mean(group2)) / pooled_std)

        return d

    def interpret_cohens_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Args:
            d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(d)

        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"

    def run_complete_analysis(
        self,
        variant_a_data: Dict[str, Any],
        variant_b_data: Dict[str, Any]
    ) -> Dict[str, StatisticalTestResult]:
        """
        Run complete statistical analysis comparing two variants
        
        Args:
            variant_a_data: {"confidence": [...], "inference_time": [...], "sentiment": {...}}
            variant_b_data: {"confidence": [...], "inference_time": [...], "sentiment": {...}}
            
        Returns:
            Dictionary of test results
        """
        results = {}

        # Compare confidence scores
        if "confidence" in variant_a_data and "confidence" in variant_b_data:
            results["confidence"] = self.compare_confidence_scores(
                variant_a_data["confidence"],
                variant_b_data["confidence"]
            )

        # Compare inference times
        if "inference_time" in variant_a_data and "inference_time" in variant_b_data:
            results["inference_time"] = self.compare_inference_times(
                variant_a_data["inference_time"],
                variant_b_data["inference_time"]
            )

        # Compare sentiment distributions
        if "sentiment" in variant_a_data and "sentiment" in variant_b_data:
            results["sentiment"] = self.compare_sentiment_distributions(
                variant_a_data["sentiment"],
                variant_b_data["sentiment"]
            )

        return results
