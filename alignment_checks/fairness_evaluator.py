import logging
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FairnessEvaluator:
    """
    Evaluates fairness of model predictions across different demographic groups.
    Includes group fairness, individual fairness, and predictive parity metrics.
    """

    def __init__(self, sensitive_attr, privileged_groups, unprivileged_groups):
        """
        Initializes the FairnessEvaluator.
        
        Args:
            sensitive_attr (str): Sensitive attribute column name.
            privileged_groups (list): Privileged group values for the attribute.
            unprivileged_groups (list): Unprivileged group values for the attribute.
        """
        self.sensitive_attr = sensitive_attr
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

    def evaluate_group_fairness(self, data: pd.DataFrame, predictions: pd.Series, labels: pd.Series):
        """
        Evaluate statistical parity and equal opportunity across groups.
        
        Args:
            data (pd.DataFrame): Dataset containing the sensitive attribute.
            predictions (pd.Series): Model predictions (binary).
            labels (pd.Series): True labels (binary).
        """
        logger.info("Evaluating group fairness...")

        group_metrics = {}

        for group in [self.privileged_groups, self.unprivileged_groups]:
            group_name = 'privileged' if group == self.privileged_groups else 'unprivileged'
            subset = data[data[self.sensitive_attr].isin(group)]

            preds_group = predictions[subset.index]
            labels_group = labels[subset.index]

            pos_rate = preds_group.mean()
            tpr = self._true_positive_rate(labels_group, preds_group)
            fpr = self._false_positive_rate(labels_group, preds_group)

            group_metrics[group_name] = {
                'positive_prediction_rate': pos_rate,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr
            }

            logger.info("Group: %s", group_name)
            logger.info("Positive Prediction Rate: %.3f", pos_rate)
            logger.info("True Positive Rate (TPR): %.3f", tpr)
            logger.info("False Positive Rate (FPR): %.3f", fpr)

        disparity = abs(group_metrics['privileged']['positive_prediction_rate'] - group_metrics['unprivileged']['positive_prediction_rate'])
        logger.info("Statistical Parity Disparity: %.3f", disparity)

        return group_metrics

    def evaluate_predictive_parity(self, data: pd.DataFrame, predictions: pd.Series, labels: pd.Series):
        """
        Evaluate predictive parity across groups.
        
        Args:
            data (pd.DataFrame): Dataset containing the sensitive attribute.
            predictions (pd.Series): Model predictions (binary).
            labels (pd.Series): True labels (binary).
        """
        logger.info("Evaluating predictive parity...")

        predictive_parity = {}

        for group in [self.privileged_groups, self.unprivileged_groups]:
            group_name = 'privileged' if group == self.privileged_groups else 'unprivileged'
            subset = data[data[self.sensitive_attr].isin(group)]

            preds_group = predictions[subset.index]
            labels_group = labels[subset.index]

            ppv = self._positive_predictive_value(labels_group, preds_group)
            predictive_parity[group_name] = {
                'positive_predictive_value': ppv
            }

            logger.info("Group: %s", group_name)
            logger.info("Positive Predictive Value (PPV): %.3f", ppv)

        disparity = abs(predictive_parity['privileged']['positive_predictive_value'] - predictive_parity['unprivileged']['positive_predictive_value'])
        logger.info("Predictive Parity Disparity: %.3f", disparity)

        return predictive_parity

    def evaluate_individual_fairness(self, data: pd.DataFrame, predictions: pd.Series, similarity_function):
        """
        Evaluate individual fairness by checking similar individuals are treated similarly.
        
        Args:
            data (pd.DataFrame): Dataset of individuals.
            predictions (pd.Series): Model predictions.
            similarity_function (callable): Function to measure similarity between two individuals.
        """
        logger.info("Evaluating individual fairness...")

        unfairness_count = 0
        total_comparisons = 0

        for i in range(len(data)):
            for j in range(i+1, len(data)):
                similarity = similarity_function(data.iloc[i], data.iloc[j])
                if similarity >= 0.9:
                    total_comparisons += 1
                    if predictions.iloc[i] != predictions.iloc[j]:
                        unfairness_count += 1

        individual_unfairness_rate = unfairness_count / total_comparisons if total_comparisons > 0 else 0.0

        logger.info("Individual Unfairness Rate: %.3f", individual_unfairness_rate)
        return individual_unfairness_rate

    @staticmethod
    def _true_positive_rate(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def _false_positive_rate(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    @staticmethod
    def _positive_predictive_value(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

if __name__ == "__main__":
    # Example run with dummy data
    import numpy as np

    data = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], size=100),
        'age': np.random.randint(18, 70, size=100)
    })

    predictions = pd.Series(np.random.randint(0, 2, size=100))
    labels = pd.Series(np.random.randint(0, 2, size=100))

    evaluator = FairnessEvaluator(
        sensitive_attr='gender',
        privileged_groups=['Male'],
        unprivileged_groups=['Female']
    )

    evaluator.evaluate_group_fairness(data, predictions, labels)
    evaluator.evaluate_predictive_parity(data, predictions, labels)
    evaluator.evaluate_individual_fairness(data, predictions, lambda x, y: 1.0 if abs(x['age'] - y['age']) < 5 else 0.0)
