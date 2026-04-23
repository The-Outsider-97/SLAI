"""
Counterfactual Fairness Metrics
Implements multi-level fairness quantification through:
- Individual counterfactual fairness (Kusner et al., 2017)
- Group-level causal disparity measures (Zhang & Bareinboim, 2018)
- Path-specific effect decomposition (Chiappa, 2019)
"""

from __future__ import annotations

import math
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from scipy.stats import wasserstein_distance
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.alignment_errors import *
from ..utils.alignment_helpers import *
from ..alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Fairness Metrics")
printer = PrettyPrinter

@dataclass(frozen=True)
class GroupConfusionStats:
    group: str
    count: int
    selection_rate: float
    true_positive_rate: float
    false_positive_rate: float
    positive_predictive_value: float
    base_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "count": self.count,
            "selection_rate": self.selection_rate,
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "positive_predictive_value": self.positive_predictive_value,
            "base_rate": self.base_rate,
        }


class CounterfactualFairness:
    """
    Multi-level counterfactual fairness assessment implementing:
    - Individual-level similarity metrics.
    - Group-level distributional comparisons.
    - Equalized-odds shift analysis across counterfactual interventions.
    - Path-specific effect decomposition for causal mediation diagnostics.

    The module is intentionally focused on reusable fairness quantification.
    It does not generate counterfactuals itself and does not own causal graph
    discovery. Instead, it consumes original and counterfactual predictions,
    along with observed data and optional graph metadata, to provide stable
    fairness metrics that higher-level auditors and monitors can rely on.
    """

    HISTORY_COLUMNS = [
        "timestamp",
        "event_id",
        "metric_family",
        "metric_name",
        "sensitive_attribute",
        "value",
        "threshold",
        "violation",
        "context_hash",
        "context",
        "metadata",
    ]

    def __init__(
        self,
        config_section_name: str = "fairness_metrics",
        *,
        memory: Optional[AlignmentMemory] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.fairness_config = get_config_section(self.config_section_name)
        self._validate_config()

        configured_sensitive = self.fairness_config.get(
            "sensitive_attributes_override",
            self.config.get("sensitive_attributes", []),
        )
        self.sensitive_attributes = list(normalize_sensitive_attributes(configured_sensitive, lowercase=False, allow_empty=True))
        self.sensitive_attrs = self.sensitive_attributes

        self.prediction_threshold = coerce_probability(
            self.fairness_config.get("prediction_threshold", 0.5),
            field_name="prediction_threshold",
        )
        self.individual_difference_threshold = coerce_probability(
            self.fairness_config.get("individual_difference_threshold", 0.1),
            field_name="individual_difference_threshold",
        )
        self.max_difference_threshold = coerce_probability(
            self.fairness_config.get("max_difference_threshold", 0.3),
            field_name="max_difference_threshold",
        )
        self.equalized_odds_gap_threshold = coerce_probability(
            self.fairness_config.get("equalized_odds_gap_threshold", 0.1),
            field_name="equalized_odds_gap_threshold",
        )
        self.disparate_impact_lower_bound = coerce_probability(
            self.fairness_config.get("disparate_impact_lower_bound", 0.8),
            field_name="disparate_impact_lower_bound",
        )
        self.min_group_size = coerce_positive_int(
            self.fairness_config.get("min_group_size", 20),
            field_name="min_group_size",
        )
        self.p_value_alpha = coerce_probability(
            self.fairness_config.get("p_value_alpha", 0.05),
            field_name="p_value_alpha",
        )
        self.bootstrap_samples = coerce_positive_int(
            self.fairness_config.get("bootstrap_samples", 250),
            field_name="bootstrap_samples",
        )
        self.path_effect_bootstrap_samples = coerce_positive_int(
            self.fairness_config.get("path_effect_bootstrap_samples", 200),
            field_name="path_effect_bootstrap_samples",
        )
        self.path_effect_confidence_level = coerce_probability(
            self.fairness_config.get("path_effect_confidence_level", 0.95),
            field_name="path_effect_confidence_level",
        )
        self.history_max_rows = coerce_positive_int(
            self.fairness_config.get("history_max_rows", 10000),
            field_name="history_max_rows",
        )
        self.enable_memory_logging = coerce_bool(
            self.fairness_config.get("enable_memory_logging", True),
            field_name="enable_memory_logging",
        )
        self.strict_memory_integration = coerce_bool(
            self.fairness_config.get("strict_memory_integration", False),
            field_name="strict_memory_integration",
        )
        self.log_distributions = coerce_bool(
            self.fairness_config.get("log_distributions", False),
            field_name="log_distributions",
        )
        self.distribution_sample_cap = coerce_positive_int(
            self.fairness_config.get("distribution_sample_cap", 256),
            field_name="distribution_sample_cap",
        )
        self.missing_group_policy = str(
            self.fairness_config.get("missing_group_policy", "skip") or "skip"
        ).strip().lower()
        self.privileged_group_policy = str(
            self.fairness_config.get("privileged_group_policy", "infer_sorted") or "infer_sorted"
        ).strip().lower()

        self.history = pd.DataFrame(columns=self.HISTORY_COLUMNS)
        self.alignment_memory = memory or AlignmentMemory()

        logger.info(
            "CounterfactualFairness initialized | sensitive_attrs=%s memory_logging=%s",
            self.sensitive_attrs,
            self.enable_memory_logging,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        try:
            ensure_mapping(
                self.fairness_config,
                self.config_section_name,
                allow_empty=True,
                error_cls=ConfigurationError,
            )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="Fairness metrics configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_individual_fairness(
        self,
        original_preds: np.ndarray,
        counterfactual_preds: np.ndarray,
        *,
        threshold: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Measure prediction stability under counterfactual changes to sensitive attributes.

        The metric family focuses on per-unit prediction drift between the original
        output and a counterfactual output where one or more sensitive attributes
        were intervened upon upstream by a causal auditor.
        """
        try:
            original = self._prepare_prediction_array(original_preds, field_name="original_preds")
            counterfactual = self._prepare_prediction_array(counterfactual_preds, field_name="counterfactual_preds")
            if original.shape != counterfactual.shape:
                raise DataValidationError(
                    "Original and counterfactual prediction arrays must have the same shape.",
                    context={
                        "original_shape": tuple(original.shape),
                        "counterfactual_shape": tuple(counterfactual.shape),
                    },
                )

            if original.size == 0:
                result = {
                    "max_difference": 0.0,
                    "mean_difference": 0.0,
                    "median_difference": 0.0,
                    "std_difference": 0.0,
                    "unfairness_rate": 0.0,
                    "wasserstein_distance": 0.0,
                    "threshold": float(threshold or self.individual_difference_threshold),
                    "sample_size": 0,
                    "violation": False,
                }
                self._record_metric(
                    metric_family="individual_fairness",
                    metric_name="mean_difference",
                    value=0.0,
                    threshold=float(threshold or self.individual_difference_threshold),
                    violation=False,
                    context=context,
                    metadata=metadata,
                )
                return result

            abs_diffs = np.abs(original - counterfactual)
            unfairness_threshold = float(threshold or self.individual_difference_threshold)
            unfairness_threshold = coerce_probability(unfairness_threshold, field_name="individual_fairness_threshold")

            result = {
                "max_difference": float(np.max(abs_diffs)),
                "mean_difference": float(np.mean(abs_diffs)),
                "median_difference": float(np.median(abs_diffs)),
                "std_difference": float(np.std(abs_diffs)),
                "unfairness_rate": float(np.mean(abs_diffs > unfairness_threshold)),
                "wasserstein_distance": float(wasserstein_distance(original, counterfactual)),
                "threshold": unfairness_threshold,
                "sample_size": int(original.size),
                "quantiles": {
                    "p90": float(np.percentile(abs_diffs, 90)),
                    "p95": float(np.percentile(abs_diffs, 95)),
                    "p99": float(np.percentile(abs_diffs, 99)),
                },
            }
            result["violation"] = bool(
                result["mean_difference"] > unfairness_threshold
                or result["max_difference"] > self.max_difference_threshold
            )

            if self.log_distributions:
                result["difference_distribution_sample"] = abs_diffs[: self.distribution_sample_cap].tolist()

            self._record_metric(
                metric_family="individual_fairness",
                metric_name="mean_difference",
                value=result["mean_difference"],
                threshold=unfairness_threshold,
                violation=result["violation"],
                context=context,
                metadata=merge_mappings(metadata, {"sample_size": result["sample_size"]}),
            )
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to compute individual counterfactual fairness.",
                context={"operation": "compute_individual_fairness"},
                metadata=normalize_metadata(metadata),
            ) from exc

    def compute_group_disparity(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        predictions: str,
        y_true: str,
        privileged_group: Optional[Union[int, str]] = None,
        unprivileged_group: Optional[Union[int, str]] = None,
        *,
        prediction_threshold: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute group disparity metrics using observed predictions.

        The returned structure preserves backward-compatible top-level disparity
        fields while adding richer summaries and per-group confusion statistics.
        """
        try:
            df = self._validate_group_input_frame(data, sensitive_attr, predictions, y_true)
            pred_threshold = float(prediction_threshold if prediction_threshold is not None else self.prediction_threshold)
            pred_threshold = coerce_probability(pred_threshold, field_name="prediction_threshold")

            preds_bin = self._binarize_predictions(df[predictions], threshold=pred_threshold)
            y_true_series = self._binarize_predictions(df[y_true], threshold=0.5)
            groups = list(df[sensitive_attr].dropna().unique())
            if len(groups) < 2:
                raise FairnessEvaluationError(
                    "At least two sensitive groups are required to compute group disparity.",
                    context={"sensitive_attr": sensitive_attr, "groups": groups},
                )

            selected_groups = self._resolve_reference_groups(groups, privileged_group, unprivileged_group)
            per_group = self._compute_per_group_stats(
                df.assign(_pred=preds_bin, _y=y_true_series),
                sensitive_attr=sensitive_attr,
                pred_col="_pred",
                y_true_col="_y",
            )
            priv_key = str(selected_groups["privileged_group"])
            unpriv_key = str(selected_groups["unprivileged_group"])
            priv_stats = per_group[priv_key]
            unpriv_stats = per_group[unpriv_key]

            spd = unpriv_stats.selection_rate - priv_stats.selection_rate
            eod = unpriv_stats.true_positive_rate - priv_stats.true_positive_rate
            aaod = 0.5 * (
                abs(unpriv_stats.false_positive_rate - priv_stats.false_positive_rate)
                + abs(unpriv_stats.true_positive_rate - priv_stats.true_positive_rate)
            )
            di_ratio = self._safe_ratio(unpriv_stats.selection_rate, priv_stats.selection_rate, default=np.nan)

            pairwise_gap_matrix = self._build_pairwise_gap_matrix(per_group)
            result = {
                "statistical_parity_difference": float(spd),
                "equal_opportunity_difference": float(eod),
                "average_abs_odds_difference": float(aaod),
                "disparate_impact_ratio": float(di_ratio) if not np.isnan(di_ratio) else np.nan,
                "tpr_privileged": float(priv_stats.true_positive_rate),
                "tpr_unprivileged": float(unpriv_stats.true_positive_rate),
                "fpr_privileged": float(priv_stats.false_positive_rate),
                "fpr_unprivileged": float(unpriv_stats.false_positive_rate),
                "privileged_group": selected_groups["privileged_group"],
                "unprivileged_group": selected_groups["unprivileged_group"],
                "prediction_threshold": pred_threshold,
                "groups": {key: stats.to_dict() for key, stats in per_group.items()},
                "pairwise_gap_matrix": pairwise_gap_matrix,
                "stat_parity_violation": abs(spd) > self.individual_difference_threshold,
                "equal_opp_violation": abs(eod) > self.equalized_odds_gap_threshold,
                "avg_odds_violation": abs(aaod) > self.equalized_odds_gap_threshold,
                "disparate_impact_violation": (not np.isnan(di_ratio)) and di_ratio < self.disparate_impact_lower_bound,
            }
            result["summary"] = {
                "n_groups": len(per_group),
                "valid_groups": list(per_group.keys()),
                "max_selection_gap": float(self._max_metric_gap(per_group, "selection_rate")),
                "max_tpr_gap": float(self._max_metric_gap(per_group, "true_positive_rate")),
                "max_fpr_gap": float(self._max_metric_gap(per_group, "false_positive_rate")),
            }

            self._record_metric(
                metric_family="group_disparity",
                metric_name=f"{sensitive_attr}_statistical_parity_difference",
                value=result["statistical_parity_difference"],
                threshold=self.individual_difference_threshold,
                violation=result["stat_parity_violation"],
                context=merge_mappings(context, {"sensitive_attr": sensitive_attr}),
                metadata=merge_mappings(metadata, {"groups": list(per_group.keys())}),
            )
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to compute group disparity.",
                context={"sensitive_attr": sensitive_attr, "predictions": predictions, "y_true": y_true},
                metadata=normalize_metadata(metadata),
            ) from exc

    def compute_equalized_odds_gap(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        original_preds_col: str,
        counterfactual_preds_col: str,
        y_true_col: str,
        *,
        prediction_threshold: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Measure how equalized-odds gaps change between original and counterfactual predictions.

        Lower gap changes indicate that the intervention does not materially worsen
        or alter pre-existing group disparities in TPR/FPR.
        """
        try:
            df = self._validate_group_input_frame(
                data,
                sensitive_attr,
                original_preds_col,
                y_true_col,
                extra_required_columns=[counterfactual_preds_col],
            )
            pred_threshold = float(prediction_threshold if prediction_threshold is not None else self.prediction_threshold)
            pred_threshold = coerce_probability(pred_threshold, field_name="prediction_threshold")

            orig_bin = self._binarize_predictions(df[original_preds_col], threshold=pred_threshold)
            cf_bin = self._binarize_predictions(df[counterfactual_preds_col], threshold=pred_threshold)
            y_true_bin = self._binarize_predictions(df[y_true_col], threshold=0.5)
            work_df = df.assign(_orig=orig_bin, _cf=cf_bin, _y=y_true_bin)

            rates = {
                "original": self._compute_per_group_stats(work_df, sensitive_attr=sensitive_attr, pred_col="_orig", y_true_col="_y"),
                "counterfactual": self._compute_per_group_stats(work_df, sensitive_attr=sensitive_attr, pred_col="_cf", y_true_col="_y"),
            }

            def _gap(group_map: Mapping[str, GroupConfusionStats], attr_name: str) -> float:
                values = [float(getattr(stats, attr_name)) for stats in group_map.values() if not np.isnan(getattr(stats, attr_name))]
                if len(values) < 2:
                    return 0.0
                return float(max(values) - min(values))

            tpr_gap_orig = _gap(rates["original"], "true_positive_rate")
            fpr_gap_orig = _gap(rates["original"], "false_positive_rate")
            tpr_gap_cf = _gap(rates["counterfactual"], "true_positive_rate")
            fpr_gap_cf = _gap(rates["counterfactual"], "false_positive_rate")

            result = {
                "tpr_gap_original": tpr_gap_orig,
                "fpr_gap_original": fpr_gap_orig,
                "tpr_gap_counterfactual": tpr_gap_cf,
                "fpr_gap_counterfactual": fpr_gap_cf,
                "tpr_gap_change": float(abs(tpr_gap_cf - tpr_gap_orig)),
                "fpr_gap_change": float(abs(fpr_gap_cf - fpr_gap_orig)),
                "threshold": self.equalized_odds_gap_threshold,
                "groups_original": {key: stats.to_dict() for key, stats in rates["original"].items()},
                "groups_counterfactual": {key: stats.to_dict() for key, stats in rates["counterfactual"].items()},
            }
            result["violation"] = bool(
                result["tpr_gap_change"] > self.equalized_odds_gap_threshold
                or result["fpr_gap_change"] > self.equalized_odds_gap_threshold
            )

            self._record_metric(
                metric_family="equalized_odds_gap",
                metric_name=f"{sensitive_attr}_equalized_odds_gap_change",
                value=max(result["tpr_gap_change"], result["fpr_gap_change"]),
                threshold=self.equalized_odds_gap_threshold,
                violation=result["violation"],
                context=merge_mappings(context, {"sensitive_attr": sensitive_attr}),
                metadata=metadata,
            )
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to compute equalized-odds gap change.",
                context={
                    "sensitive_attr": sensitive_attr,
                    "original_preds_col": original_preds_col,
                    "counterfactual_preds_col": counterfactual_preds_col,
                    "y_true_col": y_true_col,
                },
                metadata=normalize_metadata(metadata),
            ) from exc

    def compute_path_specific_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        mediators: Union[str, Sequence[str]],
        *,
        controls: Optional[Sequence[str]] = None,
        bootstrap_samples: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate direct and mediated effects using a linear path-specific decomposition.

        This uses a pragmatic mediation-style decomposition suitable for monitoring
        and fairness diagnostics:
        - Fit mediator ~ treatment + controls.
        - Fit outcome ~ treatment + mediator(s) + controls.
        - Decompose the total effect into direct and indirect components.

        The method is intentionally conservative and returns diagnostics so callers
        can understand when the decomposition rests on weak data or small samples.
        """
        try:
            ensure_instance(data, pd.DataFrame, "data", error_cls=DataValidationError)
            mediators_list = list(mediators) if not isinstance(mediators, str) else [mediators]
            mediators_list = [ensure_non_empty_string(m, "mediator", error_cls=DataValidationError) for m in mediators_list]
            control_list = [ensure_non_empty_string(c, "control", error_cls=DataValidationError) for c in (controls or [])]

            required_columns = [treatment, outcome, *mediators_list, *control_list]
            ensure_columns_present(data, required_columns, field_name="data", error_cls=DataValidationError)
            clean = data[required_columns].dropna().copy()
            if len(clean) < max(10, len(required_columns) + 2):
                raise FairnessEvaluationError(
                    "Insufficient observations to estimate path-specific effects robustly.",
                    context={"required_columns": required_columns, "n_rows": len(clean)},
                )

            mediator_models: Dict[str, RegressionResultsWrapper] = {}
            mediator_effects: Dict[str, float] = {}
            outcome_terms = [treatment, *mediators_list, *control_list]
            outcome_formula = f"{outcome} ~ {' + '.join(outcome_terms)}"
            outcome_model = smf.ols(outcome_formula, data=clean).fit()

            for mediator in mediators_list:
                mediator_terms = [treatment, *control_list]
                mediator_formula = f"{mediator} ~ {' + '.join(mediator_terms)}"
                mediator_model = smf.ols(mediator_formula, data=clean).fit()
                mediator_models[mediator] = mediator_model
                alpha_t = float(mediator_model.params.get(treatment, 0.0))
                beta_m = float(outcome_model.params.get(mediator, 0.0))
                mediator_effects[mediator] = alpha_t * beta_m

            direct_effect = float(outcome_model.params.get(treatment, 0.0))
            indirect_effect = float(sum(mediator_effects.values()))
            total_effect = direct_effect + indirect_effect

            n_boot = int(bootstrap_samples or self.path_effect_bootstrap_samples)
            boot_direct: List[float] = []
            boot_indirect: List[float] = []
            if n_boot > 1:
                rng = np.random.default_rng(42)
                for _ in range(n_boot):
                    sample_idx = rng.integers(0, len(clean), len(clean))
                    sample = clean.iloc[sample_idx]
                    boot_outcome = smf.ols(outcome_formula, data=sample).fit()
                    d_eff = float(boot_outcome.params.get(treatment, 0.0))
                    i_eff = 0.0
                    for mediator in mediators_list:
                        mediator_formula = f"{mediator} ~ {' + '.join([treatment, *control_list])}"
                        boot_mediator = smf.ols(mediator_formula, data=sample).fit()
                        i_eff += float(boot_mediator.params.get(treatment, 0.0)) * float(boot_outcome.params.get(mediator, 0.0))
                    boot_direct.append(d_eff)
                    boot_indirect.append(i_eff)

            alpha = 1.0 - self.path_effect_confidence_level
            ci_low = (alpha / 2.0) * 100.0
            ci_high = (1.0 - alpha / 2.0) * 100.0

            result = {
                "treatment": treatment,
                "outcome": outcome,
                "mediators": mediators_list,
                "controls": control_list,
                "n_obs": int(len(clean)),
                "direct_effect": direct_effect,
                "indirect_effect": indirect_effect,
                "total_effect": total_effect,
                "path_specific_effects": mediator_effects,
                "outcome_model_r_squared": float(outcome_model.rsquared),
                "outcome_model_adj_r_squared": float(outcome_model.rsquared_adj),
                "bootstrap_samples": n_boot,
            }
            if boot_direct and boot_indirect:
                result["direct_effect_ci"] = {
                    "lower": float(np.percentile(boot_direct, ci_low)),
                    "upper": float(np.percentile(boot_direct, ci_high)),
                }
                result["indirect_effect_ci"] = {
                    "lower": float(np.percentile(boot_indirect, ci_low)),
                    "upper": float(np.percentile(boot_indirect, ci_high)),
                }

            self._record_metric(
                metric_family="path_specific_effect",
                metric_name=f"{treatment}_to_{outcome}_indirect_effect",
                value=indirect_effect,
                threshold=0.0,
                violation=False,
                context=merge_mappings(context, {"treatment": treatment, "outcome": outcome}),
                metadata=merge_mappings(metadata, {"mediators": mediators_list, "controls": control_list}),
            )
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to compute path-specific effects.",
                context={"treatment": treatment, "outcome": outcome},
                metadata=normalize_metadata(metadata),
            ) from exc

    def generate_report(self) -> Dict[str, Any]:
        """Generate a compact operational report from recent fairness-metric history."""
        if self.history.empty:
            return {
                "status": "no_data",
                "history_rows": 0,
                "recent_metrics": [],
            }
        history = self.history.sort_values("timestamp")
        recent = history.tail(min(10, len(history))).copy()
        recent["timestamp"] = recent["timestamp"].astype(str)
        return {
            "status": "ok",
            "history_rows": int(len(history)),
            "metric_families": history["metric_family"].value_counts().to_dict(),
            "violation_count": int(history["violation"].sum()),
            "recent_metrics": recent.to_dict(orient="records"),
        }

    _equalized_odds_gap = compute_equalized_odds_gap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_prediction_array(self, values: Any, *, field_name: str) -> np.ndarray:
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception as exc:
            raise TypeMismatchError(
                f"'{field_name}' must be coercible to a one-dimensional numeric array.",
                context={"field": field_name, "value_type": type(values).__name__},
                cause=exc,
            ) from exc
        if np.isnan(arr).all():
            raise DataValidationError(
                f"'{field_name}' contains only NaN values.",
                context={"field": field_name},
            )
        return arr

    def _validate_group_input_frame(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        predictions: str,
        y_true: str,
        *,
        extra_required_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        ensure_instance(data, pd.DataFrame, "data", error_cls=DataValidationError)
        sensitive_attr = ensure_non_empty_string(sensitive_attr, "sensitive_attr", error_cls=DataValidationError)
        predictions = ensure_non_empty_string(predictions, "predictions", error_cls=DataValidationError)
        y_true = ensure_non_empty_string(y_true, "y_true", error_cls=DataValidationError)
        required = [sensitive_attr, predictions, y_true, *(extra_required_columns or [])]
        ensure_columns_present(data, required, field_name="data", error_cls=DataValidationError)
        if data.empty:
            raise DataValidationError("Input data must not be empty for fairness metric computation.")
        return data.copy()

    def _binarize_predictions(self, series: pd.Series, *, threshold: float) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().all():
            raise DataValidationError(
                "Prediction series could not be converted to numeric values.",
                context={"series_name": series.name},
            )
        if numeric.dropna().isin([0, 1]).all():
            return numeric.fillna(0).astype(int)
        return (numeric.fillna(0.0) >= threshold).astype(int)

    def _resolve_reference_groups(
        self,
        groups: Sequence[Any],
        privileged_group: Optional[Union[int, str]],
        unprivileged_group: Optional[Union[int, str]],
    ) -> Dict[str, Any]:
        available = list(groups)
        available_strings = [str(item) for item in available]
        if privileged_group is not None and privileged_group not in available:
            raise SensitiveAttributeError(
                "Provided privileged_group is not present in the sensitive attribute values.",
                context={"privileged_group": privileged_group, "available_groups": available_strings},
            )
        if unprivileged_group is not None and unprivileged_group not in available:
            raise SensitiveAttributeError(
                "Provided unprivileged_group is not present in the sensitive attribute values.",
                context={"unprivileged_group": unprivileged_group, "available_groups": available_strings},
            )

        if privileged_group is not None and unprivileged_group is not None:
            return {"privileged_group": privileged_group, "unprivileged_group": unprivileged_group}

        sorted_groups = sorted(available_strings)
        if len(sorted_groups) < 2:
            raise FairnessEvaluationError("At least two groups are required to infer reference groups.")

        if self.privileged_group_policy == "infer_sorted":
            inferred_unpriv = unprivileged_group if unprivileged_group is not None else sorted_groups[0]
            inferred_priv = privileged_group if privileged_group is not None else sorted_groups[-1]
        else:
            inferred_unpriv = unprivileged_group if unprivileged_group is not None else available[0]
            inferred_priv = privileged_group if privileged_group is not None else available[-1]
        return {"privileged_group": inferred_priv, "unprivileged_group": inferred_unpriv}

    def _compute_per_group_stats(
        self,
        data: pd.DataFrame,
        *,
        sensitive_attr: str,
        pred_col: str,
        y_true_col: str,
    ) -> Dict[str, GroupConfusionStats]:
        results: Dict[str, GroupConfusionStats] = {}
        grouped = data.groupby(sensitive_attr)
        for group_value, group_df in grouped:
            if len(group_df) < self.min_group_size:
                if self.missing_group_policy == "error":
                    raise FairnessEvaluationError(
                        "A sensitive group did not meet the minimum group size requirement.",
                        context={"group": str(group_value), "count": len(group_df), "min_group_size": self.min_group_size},
                    )
                if self.missing_group_policy == "skip":
                    logger.info(
                        "Skipping group '%s' for fairness metrics because count=%s < min_group_size=%s",
                        group_value,
                        len(group_df),
                        self.min_group_size,
                    )
                    continue

            selection_rate = float(group_df[pred_col].mean()) if len(group_df) else np.nan
            positives = group_df[group_df[y_true_col] == 1]
            negatives = group_df[group_df[y_true_col] == 0]
            tpr = float(positives[pred_col].mean()) if len(positives) else np.nan
            fpr = float(negatives[pred_col].mean()) if len(negatives) else np.nan
            predicted_positive = group_df[group_df[pred_col] == 1]
            ppv = float(predicted_positive[y_true_col].mean()) if len(predicted_positive) else np.nan
            base_rate = float(group_df[y_true_col].mean()) if len(group_df) else np.nan

            stats = GroupConfusionStats(
                group=str(group_value),
                count=int(len(group_df)),
                selection_rate=selection_rate,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                positive_predictive_value=ppv,
                base_rate=base_rate,
            )
            results[str(group_value)] = stats

        if len(results) < 2:
            raise FairnessEvaluationError(
                "At least two valid groups are required after group-size filtering.",
                context={"valid_groups": list(results.keys()), "min_group_size": self.min_group_size},
            )
        return results

    def _build_pairwise_gap_matrix(self, stats_map: Mapping[str, GroupConfusionStats]) -> Dict[str, Any]:
        keys = list(stats_map.keys())
        matrix: Dict[str, Any] = {}
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                left_stats = stats_map[left]
                right_stats = stats_map[right]
                pair_key = f"{left}__vs__{right}"
                matrix[pair_key] = {
                    "selection_rate_gap": float(abs(left_stats.selection_rate - right_stats.selection_rate)),
                    "tpr_gap": float(abs(left_stats.true_positive_rate - right_stats.true_positive_rate)),
                    "fpr_gap": float(abs(left_stats.false_positive_rate - right_stats.false_positive_rate)),
                    "ppv_gap": float(abs(left_stats.positive_predictive_value - right_stats.positive_predictive_value)),
                }
        return matrix

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, *, default: float = np.nan) -> float:
        if denominator is None or np.isnan(denominator) or abs(float(denominator)) < 1e-12:
            return default
        return float(numerator / denominator)

    @staticmethod
    def _max_metric_gap(stats_map: Mapping[str, GroupConfusionStats], attr_name: str) -> float:
        values = [float(getattr(stats, attr_name)) for stats in stats_map.values() if not np.isnan(getattr(stats, attr_name))]
        if len(values) < 2:
            return 0.0
        return max(values) - min(values)

    def _record_metric(
        self,
        *,
        metric_family: str,
        metric_name: str,
        value: float,
        threshold: float,
        violation: bool,
        context: Optional[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        event = build_alignment_event(
            event_type=metric_name,
            severity="high" if violation else "low",
            risk_level="high" if violation else "low",
            source="fairness_metrics",
            metadata=merge_mappings(metadata, {"metric_family": metric_family, "threshold": threshold}),
            context=context,
            payload={"value": value, "threshold": threshold, "violation": violation},
        )
        history_row = {
            "timestamp": pd.Timestamp(event["timestamp"]),
            "event_id": event["event_id"],
            "metric_family": metric_family,
            "metric_name": metric_name,
            "sensitive_attribute": str(event["context"].get("sensitive_attr", "")) or None,
            "value": float(value),
            "threshold": float(threshold),
            "violation": bool(violation),
            "context_hash": event["context_hash"],
            "context": event["context"],
            "metadata": event["metadata"],
        }
        if self.history.empty:
            self.history = pd.DataFrame([history_row])
        else:
            self.history = pd.concat([self.history, pd.DataFrame([history_row])], ignore_index=True)
        if len(self.history) > self.history_max_rows:
            self.history = self.history.tail(self.history_max_rows).reset_index(drop=True)

        if self.enable_memory_logging:
            try:
                self.alignment_memory.log_evaluation(
                    metric=normalize_metric_name(metric_name, namespace="fairness"),
                    value=float(value),
                    threshold=float(threshold),
                    context=event["context"] or {"metric_family": metric_family},
                    source="fairness_metrics",
                    tags=normalize_tags([metric_family, "fairness_metrics"]),
                    metadata=event["metadata"],
                )
            except Exception as exc:
                if self.strict_memory_integration:
                    raise wrap_alignment_exception(
                        exc,
                        target_cls=AlignmentMemoryError,
                        message="Fairness metrics failed to write to alignment memory.",
                        context={"metric_name": metric_name, "metric_family": metric_family},
                    ) from exc
                logger.warning("Fairness metrics memory logging failed: %s", exc)
        return history_row


if __name__ == "__main__":
    print("\n=== Running Fairness Metrics ===\n")
    printer.status("TEST", "Fairness Metrics initialized", "info")

    np.random.seed(42)
    size = 500
    df = pd.DataFrame({
        "A": np.random.choice([0, 1], size=size, p=[0.45, 0.55]),
        "X": np.random.normal(0, 1, size=size),
        "M": np.random.normal(0, 1, size=size),
        "Y": np.random.binomial(1, 0.5, size=size),
    })
    df["pred"] = 1 / (1 + np.exp(-(0.8 * df["X"] + 0.4 * df["M"] - 0.25 * df["A"])))
    df["pred_cf"] = 1 / (1 + np.exp(-(0.8 * df["X"] + 0.4 * df["M"] - 0.25 * (1 - df["A"]))))
    df["treatment"] = df["A"]
    df["outcome_cont"] = 0.6 * df["treatment"] + 0.5 * df["M"] + 0.3 * df["X"] + np.random.normal(0, 0.3, size=size)

    fairness = CounterfactualFairness()

    ind_fair = fairness.compute_individual_fairness(df["pred"].values, df["pred_cf"].values) # pyright: ignore[reportArgumentType]
    printer.pretty("individual_fairness", ind_fair, "success")

    group_disp = fairness.compute_group_disparity(
        data=df,
        sensitive_attr="A",
        predictions="pred",
        y_true="Y",
        privileged_group=1,
        unprivileged_group=0,
    )
    printer.pretty("group_disparity", {
        "statistical_parity_difference": group_disp["statistical_parity_difference"],
        "equal_opportunity_difference": group_disp["equal_opportunity_difference"],
        "average_abs_odds_difference": group_disp["average_abs_odds_difference"],
        "disparate_impact_ratio": group_disp["disparate_impact_ratio"],
    }, "success")

    eo_gap = fairness.compute_equalized_odds_gap(
        df.assign(original_pred=df["pred"], counterfactual_pred=df["pred_cf"]),
        "A",
        "original_pred",
        "counterfactual_pred",
        "Y",
    )
    printer.pretty("equalized_odds_gap", eo_gap, "success")

    path_effect = fairness.compute_path_specific_effect(
        data=df,
        treatment="treatment",
        outcome="outcome_cont",
        mediators=["M"],
        controls=["X"],
        bootstrap_samples=50,
    )
    printer.pretty("path_specific_effect", {
        "direct_effect": path_effect["direct_effect"],
        "indirect_effect": path_effect["indirect_effect"],
        "total_effect": path_effect["total_effect"],
        "path_specific_effects": path_effect["path_specific_effects"],
    }, "success")

    report = fairness.generate_report()
    printer.pretty("report_summary", report, "success")

    print("\n=== Test ran successfully ===\n")
