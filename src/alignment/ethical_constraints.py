import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EthicalConstraints:
    """
    Defines and enforces ethical constraints on AI agent behavior.
    Can be integrated into training loops, evaluations, and live agents.
    """

    def __init__(self, constraints_config=None):
        """
        Initializes the EthicalConstraints class.
        
        Args:
            constraints_config (dict): Optional. A dictionary defining custom ethical rules.
        """
        # Default constraints (can be loaded from a config file)
        self.constraints = constraints_config or {
            'no_harm': True,
            'respect_privacy': True,
            'avoid_bias': True,
            'avoid_discrimination': True,
            'transparency': True
        }

    def enforce(self, action_context):
        """
        Checks if a given action or decision violates any ethical constraints.
        
        Args:
            action_context (dict): Contextual data about the action/decision. Should include:
                - 'action': The action or decision taken.
                - 'target': The target user/entity.
                - 'data_used': Any data used to make the decision.
                - 'predicted_outcome': The expected outcome of the action.
        
        Returns:
            bool: True if the action is ethically compliant, False if it violates any constraint.
        """
        logger.info("Enforcing ethical constraints on action: %s", action_context.get('action'))
        
        violations = []

        # Check No Harm Constraint
        if self.constraints['no_harm']:
            if self._violates_no_harm(action_context):
                violations.append('no_harm')

        # Check Privacy Constraint
        if self.constraints['respect_privacy']:
            if self._violates_privacy(action_context):
                violations.append('respect_privacy')

        # Check Bias Constraint
        if self.constraints['avoid_bias']:
            if self._violates_bias(action_context):
                violations.append('avoid_bias')

        # Check Discrimination Constraint
        if self.constraints['avoid_discrimination']:
            if self._violates_discrimination(action_context):
                violations.append('avoid_discrimination')

        # Check Transparency Constraint
        if self.constraints['transparency']:
            if not self._meets_transparency(action_context):
                violations.append('transparency')

        if violations:
            logger.warning("Ethical constraint violations detected: %s", violations)
            return False

        logger.info("Action passed ethical constraints.")
        return True

    def _violates_no_harm(self, context):
        """Detects if an action may cause harm to users or entities."""
        predicted_outcome = context.get('predicted_outcome', {})
        if predicted_outcome.get('harm', False):
            logger.debug("Violation: Action causes harm.")
            return True
        return False

    def _violates_privacy(self, context):
        """Detects if an action breaches privacy rules."""
        data_used = context.get('data_used', {})
        if 'private' in data_used and data_used['private']:
            logger.debug("Violation: Action breaches privacy.")
            return True
        return False

    def _violates_bias(self, context):
        """Detects if an action shows biased outcomes."""
        predicted_outcome = context.get('predicted_outcome', {})
        if predicted_outcome.get('bias_detected', False):
            logger.debug("Violation: Bias detected in decision.")
            return True
        return False

    def _violates_discrimination(self, context):
        """Detects if an action unfairly discriminates against a group or individual."""
        predicted_outcome = context.get('predicted_outcome', {})
        if predicted_outcome.get('discrimination_detected', False):
            logger.debug("Violation: Discrimination detected in decision.")
            return True
        return False

    def _meets_transparency(self, context):
        """Ensures the decision/action can be explained and justified transparently."""
        explanation = context.get('explanation', '')
        if not explanation or explanation.strip() == '':
            logger.debug("Violation: No transparent explanation provided.")
            return False
        return True

    def add_constraint(self, name, rule_function):
        """
        Dynamically adds a custom ethical constraint rule.

        Args:
            name (str): Name of the new ethical rule.
            rule_function (callable): Function that implements the rule. Should return True if violated.
        """
        self.constraints[name] = True
        setattr(self, f'_violates_{name}', rule_function)
        logger.info("Added custom constraint: %s", name)

    def disable_constraint(self, name):
        """
        Disables an existing constraint by name.

        Args:
            name (str): Name of the constraint to disable.
        """
        if name in self.constraints:
            self.constraints[name] = False
            logger.info("Disabled constraint: %s", name)
        else:
            logger.warning("Constraint '%s' not found.", name)

if __name__ == "__main__":
    # Example use case
    constraints = EthicalConstraints()
    
    action_context = {
        'action': 'Serve personalized ad',
        'target': 'User123',
        'data_used': {'private': True},
        'predicted_outcome': {
            'harm': False,
            'bias_detected': False,
            'discrimination_detected': False
        },
        'explanation': 'Ad targeting based on purchase history.'
    }

    is_compliant = constraints.enforce(action_context)
    print(f"Action ethically compliant? {is_compliant}")
