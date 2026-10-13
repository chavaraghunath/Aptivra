"""Statistical Testing for Promotion Gate Regression Checks."""

import numpy as np
import scipy.stats


class StatisticalTester:
    """Performs paired statistical hypothesis testing on instance-level loss differences."""

    @staticmethod
    def evaluate_paired_regression(
        current_losses: np.ndarray,
        candidate_losses: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[bool, float, str]:
        """Tests H0: candidate loss is <= current loss vs H1: candidate loss is > current loss.

        Returns (no_significant_regression, p_value, test_name).
        """
        assert len(current_losses) == len(candidate_losses), "Array lengths must match."
        if len(current_losses) < 5:
            # Not enough samples for statistical testing
            return True, 1.0, "insufficient_samples"

        diffs = candidate_losses - current_losses

        # 1. Shapiro-Wilk test for normality
        try:
            _, p_norm = scipy.stats.shapiro(diffs)
            is_normal = p_norm >= 0.05
        except Exception:
            is_normal = False

        # 2. Execute parametric (t-test) or non-parametric (Wilcoxon) test
        if is_normal:
            res_t = scipy.stats.ttest_rel(candidate_losses, current_losses, alternative="greater")
            p_val = float(res_t.pvalue)
            test_name = "paired_t_test"
        else:
            try:
                res_w = scipy.stats.wilcoxon(diffs, alternative="greater")
                p_val = float(res_w.pvalue)
                test_name = "wilcoxon_signed_rank"
            except Exception:
                # Fallback to t-test if Wilcoxon fails
                res_t = scipy.stats.ttest_rel(
                    candidate_losses, current_losses, alternative="greater"
                )
                p_val = float(res_t.pvalue) if not np.isnan(res_t.pvalue) else 1.0
                test_name = "paired_t_test_fallback"

        # Gate Rule: Promote if p >= alpha (fail to reject H0 -> no significant regression)
        no_regression = p_val >= alpha
        return no_regression, p_val, test_name
