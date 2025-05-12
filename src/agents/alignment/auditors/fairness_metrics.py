"""
Counterfactual Fairness Metrics
Implements multi-level fairness quantification through:
- Individual counterfactual fairness (Kusner et al., 2017)
- Group-level causal disparity measures (Zhang & Bareinboim, 2018)
- Path-specific effect decomposition (Chiappa, 2019)
"""

import statsmodels.formula.api as smf
import networkx as nx
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from statsmodels.regression.linear_model import RegressionResultsWrapper
# from scipy.spatial.distance import mahalanobis
# from scipy.stats import wasserstein_distance

from src.agents.alignment.auditors.causal_model import CausalModel
from logs.logger import get_logger

logger = get_logger("Fairness Metrics")

class CounterfactualFairness:
    """
    Multi-level counterfactual fairness assessment implementing:
    - Individual-level similarity metrics
    - Group-level distributional comparisons
    - Path-specific effect decomposition
    """

    def __init__(self):
        # Consider initializing metric cache if needed, but often calculations depend on specific inputs.
        pass # No state needed for these methods currently

    def compute_individual_fairness(self,
                                  original_preds: np.ndarray,
                                  counterfactual_preds: np.ndarray) -> Dict:
        """
        Computes individual counterfactual fairness, measuring the stability
        of predictions for an individual when their sensitive attribute is changed.

        Metrics:
        - max_difference: Max absolute change in prediction.
        - mean_difference: Mean absolute change in prediction.
        - unfairness_rate: Proportion of individuals whose prediction changes above a threshold (e.g., 0.1).

        Args:
            original_preds (np.ndarray): Predictions based on original data.
            counterfactual_preds (np.ndarray): Predictions based on counterfactual data (sensitive attribute changed).

        Returns:
            Dict: Dictionary containing individual fairness metrics.
        """
        if len(original_preds) != len(counterfactual_preds):
            raise ValueError("Original and counterfactual predictions must have the same length.")
        if len(original_preds) == 0:
            return {
                'max_difference': 0.0, 'mean_difference': 0.0, 'unfairness_rate': 0.0
            }

        abs_diffs = np.abs(original_preds - counterfactual_preds)

        # Basic individual fairness metrics
        metrics = {
            'max_difference': float(np.max(abs_diffs)),
            'mean_difference': float(np.mean(abs_diffs)),
            'unfairness_rate': float(np.mean(abs_diffs > 0.1)) # Threshold can be adjusted
        }
        return metrics


    def compute_group_disparity(self,
                              data: pd.DataFrame,
                              sensitive_attr: str,
                              predictions: str,
                              y_true: str,
                              privileged_group: Union[int, str],
                              unprivileged_group: Union[int, str]) -> Dict:
        """
        Computes standard group fairness metrics based on observed predictions.

        Args:
            data (pd.DataFrame): DataFrame containing sensitive attribute, predictions, and true labels.
            sensitive_attr (str): Column name of the sensitive attribute.
            predictions (str): Column name of the model predictions (binary 0/1 or probabilities).
            y_true (str): Column name of the ground truth labels (binary 0/1).
            privileged_group (Union[int, str]): Value representing the privileged group in sensitive_attr.
            unprivileged_group (Union[int, str]): Value representing the unprivileged group in sensitive_attr.

        Returns:
            Dict: Dictionary containing group fairness metrics like Statistical Parity Difference,
                  Equal Opportunity Difference, Average Absolute Odds Difference.
        """
        if sensitive_attr not in data.columns: raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found.")
        if predictions not in data.columns: raise ValueError(f"Predictions column '{predictions}' not found.")
        if y_true not in data.columns: raise ValueError(f"True labels column '{y_true}' not found.")

        # Binarize predictions if they are probabilities (common 0.5 threshold)
        if data[predictions].dtype != 'int' and data[predictions].between(0, 1).all():
            logger.info(f"Binarizing predictions '{predictions}' using 0.5 threshold.")
            preds_bin = (data[predictions] > 0.5).astype(int)
        else:
            preds_bin = data[predictions] # Assume already binary

        df_priv = data[data[sensitive_attr] == privileged_group]
        df_unpriv = data[data[sensitive_attr] == unprivileged_group]

        if df_priv.empty or df_unpriv.empty:
            logger.warning("One or both groups are empty. Cannot compute group disparity.")
            return {'statistical_parity_difference': np.nan, 'equal_opportunity_difference': np.nan, 'average_abs_odds_difference': np.nan}

        # --- Calculations ---
        # Statistical Parity Difference (SPD) = P(Y_hat=1 | A=unpriv) - P(Y_hat=1 | A=priv)
        rate_priv = preds_bin.loc[df_priv.index].mean()
        rate_unpriv = preds_bin.loc[df_unpriv.index].mean()
        spd = rate_unpriv - rate_priv

        # True Positive Rate (TPR) = P(Y_hat=1 | Y=1, A=a)
        # False Positive Rate (FPR) = P(Y_hat=1 | Y=0, A=a)
        tpr_priv = preds_bin.loc[df_priv[df_priv[y_true] == 1].index].mean() if (df_priv[y_true] == 1).any() else 0
        tpr_unpriv = preds_bin.loc[df_unpriv[df_unpriv[y_true] == 1].index].mean() if (df_unpriv[y_true] == 1).any() else 0
        fpr_priv = preds_bin.loc[df_priv[df_priv[y_true] == 0].index].mean() if (df_priv[y_true] == 0).any() else 0
        fpr_unpriv = preds_bin.loc[df_unpriv[df_unpriv[y_true] == 0].index].mean() if (df_unpriv[y_true] == 0).any() else 0

        # Equal Opportunity Difference (EOD) = TPR(unpriv) - TPR(priv)
        eod = tpr_unpriv - tpr_priv

        # Average Absolute Odds Difference (AAOD) = 0.5 * [ |FPR(unpriv)-FPR(priv)| + |TPR(unpriv)-TPR(priv)| ]
        aaod = 0.5 * (abs(fpr_unpriv - fpr_priv) + abs(tpr_unpriv - tpr_priv))

        return {
            'statistical_parity_difference': float(spd),
            'equal_opportunity_difference': float(eod),
            'average_abs_odds_difference': float(aaod),
            'tpr_privileged': float(tpr_priv),
            'tpr_unprivileged': float(tpr_unpriv),
            'fpr_privileged': float(fpr_priv),
            'fpr_unprivileged': float(fpr_unpriv)
        }


    def _equalized_odds_gap(self,
                            causal_model: CausalModel,
                            data: pd.DataFrame,
                            sensitive_attr: str,
                            original_preds_col: str,
                            counterfactual_preds_col: str,
                            y_true_col: str) -> Dict[str, float]:
        r"""
        Calculates the Equalized Odds gap between original and counterfactual predictions.
        Equalized odds requires that the prediction $\hat{Y}$ is independent of the
        sensitive attribute $A$, conditional on the true outcome $Y$.
        $P(\hat{Y}=1 | A=a, Y=y) = P(\hat{Y}=1 | A=a', Y=y)$ for all $a, a'$ and $y \in \{0, 1\}$.
        This translates to requiring both TPR and FPR to be equal across groups.

        This function measures the *change* in TPR and FPR disparities between the
        original predictions and counterfactual predictions (where A was hypothetically changed).

        Args:
            data (pd.DataFrame): DataFrame containing sensitive attr, true labels,
                                 original predictions, and counterfactual predictions.
            sensitive_attr (str): Column name of the sensitive attribute.
            original_preds_col (str): Column name for original predictions.
            counterfactual_preds_col (str): Column name for counterfactual predictions.
            y_true_col (str): Column name for ground truth labels.

        Returns:
            Dict[str, float]: Dictionary with 'tpr_gap_change' and 'fpr_gap_change'.
                              These represent the absolute difference in TPR/FPR gaps
                              between the original and counterfactual scenarios.
                              Lower values indicate the counterfactual change did not
                              exacerbate existing equalized odds violations.
        """
        if not all(col in data.columns for col in [sensitive_attr, original_preds_col, counterfactual_preds_col, y_true_col]):
            raise ValueError("One or more required columns are missing from the DataFrame.")

        groups = data[sensitive_attr].unique()
        if len(groups) < 2:
            logger.warning("Only one group found. Cannot compute equalized odds gaps between groups.")
            return {'tpr_gap_change': 0.0, 'fpr_gap_change': 0.0}

        rates = {'original': {'tpr': {}, 'fpr': {}}, 'counterfactual': {'tpr': {}, 'fpr': {}}}

        # Binarize predictions if needed
        preds_orig_bin = (data[original_preds_col] > 0.5).astype(int) if data[original_preds_col].between(0, 1).all() else data[original_preds_col]
        preds_cf_bin = (data[counterfactual_preds_col] > 0.5).astype(int) if data[counterfactual_preds_col].between(0, 1).all() else data[counterfactual_preds_col]

        for group_val in groups:
            group_filter = (data[sensitive_attr] == group_val)
            y_true_group = data.loc[group_filter, y_true_col]

            # --- Original Predictions ---
            preds_orig_group = preds_orig_bin[group_filter]
            # TPR = P(Y_hat=1 | Y=1, A=group_val)
            true_positives_filter_orig = group_filter & (data[y_true_col] == 1)
            actual_positives_count_orig = true_positives_filter_orig.sum()
            if actual_positives_count_orig > 0:
                rates['original']['tpr'][group_val] = preds_orig_bin[true_positives_filter_orig].mean()
            else:
                rates['original']['tpr'][group_val] = np.nan # Avoid division by zero

            # FPR = P(Y_hat=1 | Y=0, A=group_val)
            true_negatives_filter_orig = group_filter & (data[y_true_col] == 0)
            actual_negatives_count_orig = true_negatives_filter_orig.sum()
            if actual_negatives_count_orig > 0:
                 rates['original']['fpr'][group_val] = preds_orig_bin[true_negatives_filter_orig].mean()
            else:
                rates['original']['fpr'][group_val] = np.nan

            # --- Counterfactual Predictions ---
            preds_cf_group = preds_cf_bin[group_filter]
             # TPR = P(Y_hat_cf=1 | Y=1, A=group_val) - Note A is original, Y_hat is CF
            true_positives_filter_cf = group_filter & (data[y_true_col] == 1) # Same filter as original
            actual_positives_count_cf = true_positives_filter_cf.sum() # Same count
            if actual_positives_count_cf > 0:
                rates['counterfactual']['tpr'][group_val] = preds_cf_bin[true_positives_filter_cf].mean()
            else:
                rates['counterfactual']['tpr'][group_val] = np.nan

            # FPR = P(Y_hat_cf=1 | Y=0, A=group_val)
            true_negatives_filter_cf = group_filter & (data[y_true_col] == 0) # Same filter as original
            actual_negatives_count_cf = true_negatives_filter_cf.sum() # Same count
            if actual_negatives_count_cf > 0:
                rates['counterfactual']['fpr'][group_val] = preds_cf_bin[true_negatives_filter_cf].mean()
            else:
                 rates['counterfactual']['fpr'][group_val] = np.nan


        # Calculate gaps (difference between max and min rate across groups)
        def calculate_gap(rate_dict):
            valid_rates = [r for r in rate_dict.values() if not np.isnan(r)]
            if len(valid_rates) < 2: return 0.0
            return max(valid_rates) - min(valid_rates)

        tpr_gap_orig = calculate_gap(rates['original']['tpr'])
        fpr_gap_orig = calculate_gap(rates['original']['fpr'])
        tpr_gap_cf = calculate_gap(rates['counterfactual']['tpr'])
        fpr_gap_cf = calculate_gap(rates['counterfactual']['fpr'])

        # Change in gaps
        tpr_gap_change = abs(tpr_gap_cf - tpr_gap_orig)
        fpr_gap_change = abs(fpr_gap_cf - fpr_gap_orig)

        return {'tpr_gap_change': tpr_gap_change, 'fpr_gap_change': fpr_gap_change}


    def path_specific_effects(self,
                            causal_model: CausalModel,
                            data: pd.DataFrame,
                            sensitive_attr: str,
                            outcome: str,
                            sensitive_val_1: Union[int, float],
                            sensitive_val_0: Union[int, float],
                            mediator_paths: Optional[List[List[str]]] = None) -> Dict[str, float]:
        r"""
        Decompose the total effect of a sensitive attribute on the outcome into
        path-specific effects, particularly Natural Direct Effect (NDE) and
        Natural Indirect Effect (NIE).

        Requires a fitted CausalModel object. Assumes binary sensitive attribute for NDE/NIE calc.

        Definitions (Pearl, 2001):
        - Total Effect (TE): $TE = E[Y(A=a_1) - Y(A=a_0)]$
          How much does the outcome change overall when the sensitive attribute changes?
        - Natural Direct Effect (NDE): $NDE = E[Y(A=a_1, M(A=a_0)) - Y(A=a_0, M(A=a_0))]$
          How much would the outcome change if we changed the sensitive attribute from $a_0$ to $a_1$,
          but kept the mediators $M$ as they would have been if $A=a_0$? Measures effect not through M.
        - Natural Indirect Effect (NIE): $NIE = E[Y(A=a_0, M(A=a_1)) - Y(A=a_0, M(A=a_0))]$
          How much would the outcome change if we kept the sensitive attribute fixed at $a_0$,
          but changed the mediators $M$ to what they would have been if $A=a_1$? Measures effect only through M.

        For linear systems, $TE = NDE + NIE$. For non-linear systems, this is approximate.

        Args:
            causal_model (CausalModel): An instance of the CausalModel class with a graph and SEMs.
            data (pd.DataFrame): The dataset.
            sensitive_attr (str): The name of the sensitive attribute column (A).
            outcome (str): The name of the outcome column (Y).
            sensitive_val_1: The 'treated' or target value of the sensitive attribute.
            sensitive_val_0: The 'control' or reference value of the sensitive attribute.
            mediator_paths (Optional[List[List[str]]]): Specific mediating paths A->M->Y to analyze.
                                                       If None, attempts to identify mediators from graph.

        Returns:
            Dict[str, float]: Dictionary containing TE, NDE, NIE estimates.
        """
        if not isinstance(causal_model, CausalModel):
            # Check if it's the placeholder due to import error
            try:
                causal_model.compute_counterfactual() # Will raise error if it's the placeholder
            except NotImplementedError:
                logger.error("CausalModel not available due to import error. Cannot compute path-specific effects.")
                return {'TE': np.nan, 'NDE': np.nan, 'NIE': np.nan}
            except Exception: # Catch other errors if compute_counterfactual exists but fails differently
                pass # Proceed if it seems like a real CausalModel instance

            # If not the placeholder, but still not a CausalModel instance
            if not hasattr(causal_model, 'compute_counterfactual') or not hasattr(causal_model, 'graph'):
                raise TypeError("causal_model must be an instance of the CausalModel class.")


        logger.info(f"Calculating path-specific effects: {sensitive_attr} -> {outcome}")

        # --- Identify Mediators (M) ---
        # Mediators are nodes on directed paths from A to Y, excluding A and Y.
        mediators = set()
        try:
            # Ensure graph attribute exists and is a DiGraph
            if not hasattr(causal_model, 'graph') or not isinstance(causal_model.graph, nx.DiGraph):
                raise AttributeError("CausalModel instance does not have a valid 'graph' attribute.")

            if nx.has_path(causal_model.graph, sensitive_attr, outcome):
                for path in nx.all_simple_paths(causal_model.graph, source=sensitive_attr, target=outcome):
                    if len(path) > 2: # Path must have at least one mediator
                        mediators.update(path[1:-1]) # Add nodes between A and Y
        except nx.NodeNotFound:
            logger.error(f"Sensitive attribute '{sensitive_attr}' or outcome '{outcome}' not found in causal graph.")
            return {'TE': np.nan, 'NDE': np.nan, 'NIE': np.nan}
        except AttributeError as e:
            logger.error(f"Error accessing causal model graph: {e}")
            return {'TE': np.nan, 'NDE': np.nan, 'NIE': np.nan}


        if not mediators:
            logger.warning(f"No mediating paths found between {sensitive_attr} and {outcome}. NDE will equal TE, NIE will be 0.")
            # Calculate TE only in this case
            try:
                y_a1 = causal_model.compute_counterfactual(intervention={sensitive_attr: sensitive_val_1})[outcome]
                y_a0 = causal_model.compute_counterfactual(intervention={sensitive_attr: sensitive_val_0})[outcome]
                te = (y_a1 - y_a0).mean()
                return {'TE': float(te), 'NDE': float(te), 'NIE': 0.0}
            except Exception as e:
                logger.error(f"Failed to compute total effect: {e}")
                return {'TE': np.nan, 'NDE': np.nan, 'NIE': np.nan}

        logger.info(f"Identified potential mediators: {mediators}")
        list_mediators = list(mediators)

        # --- Compute Counterfactuals for NDE/NIE ---
        # We need E[Y(a1)], E[Y(a0)], E[Y(a1, M(a0))], E[Y(a0, M(a1))]
        try:
            # E[Y(a1)] and E[Y(a0)] -> Needed for TE
            cf_data_a1 = causal_model.compute_counterfactual(intervention={sensitive_attr: sensitive_val_1})
            cf_data_a0 = causal_model.compute_counterfactual(intervention={sensitive_attr: sensitive_val_0})
            y_a1_mean = cf_data_a1[outcome].mean()
            y_a0_mean = cf_data_a0[outcome].mean()
            te = y_a1_mean - y_a0_mean
            # For NDE/NIE, we need nested counterfactuals. This requires careful application of SEMs.
            # Method: Simulate M(a0) and M(a1), then plug into Y's SEM under do(a).
            # 1. Get counterfactual mediator values M(a0) and M(a1)
            #    These are the values of mediators from the cf_data computed above.
            M_a0_vals = cf_data_a0[list_mediators]
            M_a1_vals = cf_data_a1[list_mediators]

            # 2. Compute Y(a1, M(a0)) and Y(a0, M(a1))
            #    This involves predicting Y using its SEM, setting A=a and M=M_a'
            #    while other parents of Y are determined by A=a.

            # Get Y's structural equation
            sems = causal_model._get_structural_equations() # Use the internal getter
            y_model_info = sems.get(outcome)
            y_parents = list(causal_model.graph.predecessors(outcome))
            other_parents = [p for p in y_parents if p != sensitive_attr and p not in list_mediators]
            # Use the correct type from statsmodels.regression.linear_model
            if y_model_info is None or not isinstance(y_model_info, RegressionResultsWrapper): # CORRECTED Check type
                logger.error(f"Could not retrieve valid structural equation for outcome '{outcome}'. Cannot compute NDE/NIE.")
                # Return TE if available, otherwise NaN
                te_val = np.nan # Default if TE wasn't calculated before this check
                if 'te' in locals():
                    try:
                        te_val = float(te)
                    except (NameError, TypeError):
                        te_val = np.nan
                return {'TE': te_val, 'NDE': np.nan, 'NIE': np.nan}

            # 2. Replace the predict_y function definition (around line 372) with the corrected version:
            # Function to predict Y given specific A, M, and other parent values
            def predict_y(a_val, m_vals_df, parent_data_df):
                """Predicts outcome using the structural equation for Y."""
                # Construct the input DataFrame for prediction, using index from mediator values
                pred_input = pd.DataFrame(index=m_vals_df.index)
                pred_input[sensitive_attr] = a_val

                # Add mediator values
                for med in list_mediators:
                    if med in m_vals_df.columns:
                        pred_input[med] = m_vals_df[med]
                    else:
                        logger.warning(f"Mediator {med} not found in m_vals_df during predict_y for outcome {outcome}.")
                        # Need a fallback value if mediator is missing
                        pred_input[med] = 0 # Or data[med].mean()? Requires access/passing 'data'

                # Add other parent values (non-sensitive, non-mediator parents of Y)
                for op in other_parents:
                    if op in parent_data_df.columns:
                        pred_input[op] = parent_data_df[op]
                    else:
                        logger.warning(f"Other parent {op} of {outcome} not found in parent_data_df during predict_y.")
                        # Fallback to mean from original data - use with caution
                        if op in data.columns:
                            pred_input[op] = data[op].mean()
                        else:
                            logger.error(f"Other parent {op} not found in original data either!")
                            pred_input[op] = 0 # Last resort fallback

                # Align with model's expected inputs for prediction
                try:
                    y_model = y_model_info # Already checked isinstance above
                    has_intercept = 'const' in y_model.model.exog_names
                    if has_intercept:
                        # Add constant term if model expects it
                        pred_input['const'] = 1.0

                    # Reindex to match the exact columns the model was trained on
                    pred_input_aligned = pred_input.reindex(columns=y_model.model.exog_names, fill_value=0)

                    # Return predictions
                    return y_model.predict(pred_input_aligned)
                except Exception as e:
                    logger.error(f"Error during predict_y for outcome {outcome} using parents {y_parents}: {e}")
                    # Return NaNs matching the expected output index length
                    return pd.Series(np.nan, index=pred_input.index)


            # 3. Ensure the calls to predict_y use the potentially NaN series correctly (around lines 395-403):
            # Predict Y(a1, M(a0))
            y_pred_a1_ma0 = predict_y(sensitive_val_1, M_a0_vals, cf_data_a1)

            # Predict Y(a0, M(a1))
            y_pred_a0_ma1 = predict_y(sensitive_val_0, M_a1_vals, cf_data_a0)

            # Predict Y(a0, M(a0)) - this is just E[Y(a0)] calculated before
            # Ensure y_a0_mean is valid before using
            y_a0_ma0_mean = y_a0_mean if not np.isnan(y_a0_mean) else np.nan

            # Calculate NDE and NIE, handling potential NaNs from prediction failures
            nde = np.nanmean(y_pred_a1_ma0) - y_a0_ma0_mean if y_pred_a1_ma0 is not None else np.nan
            nie = np.nanmean(y_pred_a0_ma1) - y_a0_ma0_mean if y_pred_a0_ma1 is not None else np.nan

            logger.info(f"Path-specific effects calculated: TE={te:.4f}, NDE={nde:.4f}, NIE={nie:.4f}")
            # Sanity check: NDE + NIE should approximate TE for linear models
            if abs((nde + nie) - te) > 0.01: # Allow small tolerance for float errors / non-linearity
                logger.warning(f"NDE ({nde:.4f}) + NIE ({nie:.4f}) = {nde+nie:.4f}, which differs significantly from TE ({te:.4f}). May indicate non-linearities or issues.")

            return {'TE': float(te), 'NDE': float(nde), 'NIE': float(nie)}

        except Exception as e:
            logger.error(f"Failed during path-specific effect calculation: {e}", exc_info=True) # Log traceback
            # Try to return TE if calculated
            try:
                te_val = float(te) if 'te' in locals() else np.nan
            except NameError:
                te_val = np.nan
            return {'TE': te_val, 'NDE': np.nan, 'NIE': np.nan}

if __name__ == "__main__":
    # Synthetic test data
    np.random.seed(42)
    size = 100
    df = pd.DataFrame({
        'A': np.random.choice([0, 1], size=size),  # Sensitive attribute
        'X': np.random.normal(0, 1, size=size),    # Mediator
        'Y': np.random.binomial(1, 0.5, size=size) # Outcome
    })
    df['pred'] = df['A'] * 0.3 + df['X'] * 0.5 + np.random.normal(0, 0.1, size=size)
    df['pred_cf'] = (1 - df['A']) * 0.3 + df['X'] * 0.5 + np.random.normal(0, 0.1, size=size)

    # Initialize fairness metrics
    fairness = CounterfactualFairness()

    # Test compute_individual_fairness
    ind_fair = fairness.compute_individual_fairness(df['pred'].values, df['pred_cf'].values)
    print("Individual Fairness:", ind_fair)

    # Test compute_group_disparity
    group_disp = fairness.compute_group_disparity(df, 'A', 'pred', 'Y', 1, 0)
    print("Group Disparity:", group_disp)

    # Test path-specific effects (simple linear graph: A -> X -> Y)
    G = nx.DiGraph()
    G.add_edges_from([('A', 'X'), ('X', 'Y')])
    model = CausalModel(graph=G, data=df)
    p_effects = fairness.path_specific_effects(model, df, 'A', 'Y', 1, 0)
    print("Path-Specific Effects:", p_effects)
    # Test equalized odds gap
    eo_gap = fairness._equalized_odds_gap(model,
                                          df.assign(original_pred=df['pred'], counterfactual_pred=df['pred_cf']),
                                          'A', 'original_pred', 'counterfactual_pred', 'Y')
    print("Equalized Odds Gap Change:", eo_gap)
