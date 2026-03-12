
from typing import List, Dict, Any, Tuple, Optional, Set

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Deductive")
printer = PrettyPrinter

class ReasoningDeductive(BaseReasoning):
    """
    Implements deductive reasoning: Deriving specific conclusions from general premises
    Process:
    1. Validate premises and rules
    2. Apply rules of inference to premises
    3. Derive intermediate conclusions
    4. Reach final conclusion with logical certainty
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.contradiction_threshold = self.config.get('contradiction_threshold')

        self.deductive_config = get_config_section("reasoning_deductive")
        self.max_steps = self.deductive_config.get('max_steps')
        self.certainty_threshold = self.deductive_config.get('certainty_threshold')
        self.enable_fallacy_check = self.deductive_config.get('enable_fallacy_check')
        self.rule_priority = self.deductive_config.get('rule_priority', [])

    def perform_reasoning(self, premises: List[str], hypothesis: str, context: dict = None) -> Dict[str, Any]:
        """
        Perform deductive reasoning from premises to hypothesis validation
        Args:
            premises: List of assumed true statements
            hypothesis: Statement to prove or disprove
            context: Additional context for reasoning
        Returns:
            Proof structure with validation result and certainty
        """
        self.log_step("Starting deductive reasoning process")
        context = context or {}
        
        # Step 1: Validate premises
        valid_premises = self._validate_premises(premises, context)
        if not valid_premises:
            return self._format_result(False, 0.0, [], premises, hypothesis, "Invalid premises")
        
        # Step 2: Apply inference rules
        proof_steps = self._apply_inference_rules(valid_premises, context)
        
        # Step 3: Evaluate hypothesis
        evaluation = self._evaluate_hypothesis(hypothesis, proof_steps, context)
        
        # Step 4: Check for contradictions
        contradiction_analysis = self._check_contradictions(proof_steps, hypothesis, context)
        
        # Step 5: Determine logical certainty
        is_proven, certainty = self._determine_certainty(evaluation, contradiction_analysis)
        
        # Format and return results
        return self._format_result(
            is_proven, 
            certainty, 
            proof_steps, 
            valid_premises, 
            hypothesis, 
            contradiction_analysis
        )

    def _validate_premises(self, premises: List[str], context: Dict) -> List[str]:
        """Validate premises for logical consistency and truthfulness"""
        valid_premises = []
        
        # Check for internal contradictions
        contradictions = self.identify_contradictions(premises)
        if contradictions:
            self.log_step(f"Found contradictions in premises: {contradictions}", "warning")
            return []
        
        # Validate each premise
        for premise in premises:
            if self._is_valid_premise(premise, context):
                valid_premises.append(premise)
            else:
                self.log_step(f"Invalid premise rejected: {premise}", "warning")
        
        return valid_premises

    def identify_contradictions(self, premises: List[str]) -> List[Tuple[str, str, str]]:
        """
        Identify logical contradictions among premises using deductive analysis.
        Leverages base logic and adds contradiction threshold filtering and advanced negation.
        
        Args:
            premises: List of logical premises to validate.
        
        Returns:
            List of contradiction tuples (statement1, statement2, contradiction_type)
        """
        self.log_step("Analyzing premises for contradictions...")
    
        # Step 1: Use base logic detection
        contradictions = super().identify_contradictions(premises)
        
        if not contradictions:
            self.log_step("No direct contradictions found among premises.")
            return []
    
        # Step 2: Apply contradiction filtering (e.g., threshold)
        filtered = []
        for pair in contradictions:
            stmt1, stmt2, conflict_type = pair
    
            # Optional: Only keep contradictions that violate a threshold
            if self.contradiction_threshold and conflict_type == "direct_negation":
                if len(stmt1) > 10 and len(stmt2) > 10:  # heuristic filtering
                    filtered.append(pair)
            else:
                filtered.append(pair)
        
        # Step 3: Log result and return
        if filtered:
            self.log_step(f"Detected {len(filtered)} contradiction(s)", level="warning")
            for c in filtered:
                self.log_step(f"Conflict: {c[0]} vs {c[1]} ({c[2]})", level="warning")
        else:
            self.log_step("All contradictions were filtered out by heuristics.")
    
        return filtered

    def _is_valid_premise(self, premise: str, context: Dict) -> bool:
        """Determine if a premise is valid using context and evidence"""
        # Check against known truths in context
        known_falsehoods = context.get("known_falsehoods", [])
        if premise in known_falsehoods:
            return False
        
        # Check with evidence sources
        evidence = context.get("evidence", [])
        for item in evidence:
            if item.get("content") == premise and item.get("confidence", 0) < self.certainty_threshold:
                return False
        
        # Apply formal logic validation
        return self._formal_validation(premise)

    def _formal_validation(self, statement: str) -> bool:
        """Validate statement structure using formal logic rules"""
        # Simple validation: Must have subject-predicate structure
        if " is " not in statement and " are " not in statement and " has " not in statement:
            return False
        return True

    def _apply_inference_rules(self, premises: List[str], context: Dict) -> List[Dict]:
        """Apply inference rules to premises to derive conclusions"""
        proof_steps = []
        current_statements = premises.copy()
        step_count = 0
        
        while step_count < self.max_steps:
            new_conclusions = []
            
            # Apply rules in priority order
            for rule_name in self.rule_priority:
                rule_method = getattr(self, f"_apply_{rule_name}", None)
                if rule_method:
                    conclusions = rule_method(current_statements, context)
                    for conclusion in conclusions:
                        if conclusion not in current_statements:
                            new_conclusions.append(conclusion)
                            proof_steps.append({
                                "step": step_count,
                                "rule": rule_name,
                                "input": current_statements.copy(),
                                "output": conclusion
                            })
                            step_count += 1
                            if step_count >= self.max_steps:
                                break
            
            # Add new conclusions to current statements
            current_statements.extend(new_conclusions)
            
            # Stop if no new conclusions
            if not new_conclusions:
                break
        
        return proof_steps

    def _apply_modus_ponens(self, statements: List[str], context: Dict) -> List[str]:
        """Apply Modus Ponens: If P implies Q, and P is true, then Q is true"""
        conclusions = []
        implication_patterns = ["implies", "->", "therefore", "consequently"]
        
        for stmt in statements:
            for pattern in implication_patterns:
                if pattern in stmt:
                    parts = stmt.split(pattern)
                    if len(parts) == 2:
                        antecedent = parts[0].strip()
                        consequent = parts[1].strip()
                        
                        # Check if antecedent exists in statements
                        if antecedent in statements:
                            conclusions.append(consequent)
        
        return conclusions

    def _apply_modus_tollens(self, statements: List[str], context: Dict) -> List[str]:
        """Apply Modus Tollens: If P implies Q, and Q is false, then P is false"""
        conclusions = []
        implication_patterns = ["implies", "->", "therefore", "consequently"]
        negation_words = ["not", "false", "no", "never"]
        
        for stmt in statements:
            for pattern in implication_patterns:
                if pattern in stmt:
                    parts = stmt.split(pattern)
                    if len(parts) == 2:
                        antecedent = parts[0].strip()
                        consequent = parts[1].strip()
                        
                        # Check if negation of consequent exists
                        for negation in negation_words:
                            negated_consequent = f"{negation} {consequent}" if not consequent.startswith(negation) else consequent.replace(negation, "").strip()
                            
                            if negated_consequent in statements:
                                # Negate antecedent
                                negated_antecedent = f"not {antecedent}"
                                conclusions.append(negated_antecedent)
        
        return conclusions

    def _apply_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        """Apply categorical syllogism: If all A are B, and all B are C, then all A are C"""
        conclusions = []
        universal_patterns = ["all", "every", "each"]
        
        # Find universal affirmative statements
        universal_statements = []
        for stmt in statements:
            for pattern in universal_patterns:
                if pattern in stmt:
                    universal_statements.append(stmt)
        
        # Check for syllogism patterns
        for i in range(len(universal_statements)):
            for j in range(i+1, len(universal_statements)):
                stmt1 = universal_statements[i]
                stmt2 = universal_statements[j]
                
                # Extract subject and predicate
                parts1 = stmt1.split(" are ")
                parts2 = stmt2.split(" are ")
                
                if len(parts1) == 2 and len(parts2) == 2:
                    subject1 = parts1[0].replace("all ", "").replace("every ", "").replace("each ", "").strip()
                    predicate1 = parts1[1].strip()
                    
                    subject2 = parts2[0].replace("all ", "").replace("every ", "").replace("each ", "").strip()
                    predicate2 = parts2[1].strip()
                    
                    # Check for middle term connection
                    if predicate1 == subject2:
                        conclusion = f"all {subject1} are {predicate2}"
                        conclusions.append(conclusion)
                    elif predicate2 == subject1:
                        conclusion = f"all {subject2} are {predicate1}"
                        conclusions.append(conclusion)
        
        return conclusions

    def _apply_disjunctive_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        """Apply Disjunctive Syllogism: If P or Q is true, and P is false, then Q is true"""
        conclusions = []
        disjunction_patterns = [" or ", " | ", " either "]
        negation_words = ["not", "false", "no", "never"]
        
        for stmt in statements:
            for pattern in disjunction_patterns:
                if pattern in stmt:
                    options = [opt.strip() for opt in stmt.split(pattern)]
                    
                    # Find which option is negated in statements
                    for option in options:
                        for negation in negation_words:
                            negated_option = f"{negation} {option}" if not option.startswith(negation) else option.replace(negation, "").strip()
                            
                            if negated_option in statements:
                                # The other option must be true
                                other_options = [opt for opt in options if opt != option]
                                conclusions.extend(other_options)
        
        return conclusions

    def _apply_hypothetical_syllogism(self, statements: List[str], context: Dict) -> List[str]:
        """Apply Hypothetical Syllogism: If P implies Q, and Q implies R, then P implies R"""
        conclusions = []
        implication_patterns = ["implies", "->", "therefore", "consequently"]
        implications = []
        
        # Collect all implications
        for stmt in statements:
            for pattern in implication_patterns:
                if pattern in stmt:
                    implications.append(stmt)
        
        # Chain implications
        for i in range(len(implications)):
            for j in range(len(implications)):
                if i != j:
                    parts1 = implications[i].split(" implies ")
                    parts2 = implications[j].split(" implies ")
                    
                    if len(parts1) == 2 and len(parts2) == 2:
                        antecedent1 = parts1[0].strip()
                        consequent1 = parts1[1].strip()
                        
                        antecedent2 = parts2[0].strip()
                        consequent2 = parts2[1].strip()
                        
                        # Check for connection
                        if consequent1 == antecedent2:
                            conclusion = f"{antecedent1} implies {consequent2}"
                            conclusions.append(conclusion)
        
        return conclusions

    def _evaluate_hypothesis(self, hypothesis: str, proof_steps: List[Dict], context: Dict) -> Dict[str, Any]:
        """Evaluate whether the hypothesis is proven by the deduction"""
        # Check if hypothesis appears in any conclusion
        direct_proof = any(
            step["output"] == hypothesis for step in proof_steps
        )
        
        # Check if negation of hypothesis is proven
        negation_proof = any(
            step["output"] == f"not {hypothesis}" or 
            step["output"] == f"false {hypothesis}" 
            for step in proof_steps
        )
        
        # Check if hypothesis can be derived from conclusions
        derived_proof = False
        all_conclusions = [step["output"] for step in proof_steps]
        if self._can_derive(hypothesis, all_conclusions, context):
            derived_proof = True
        
        # Calculate proof strength
        proof_strength = 0.0
        if direct_proof:
            proof_strength = 0.9
        elif derived_proof:
            proof_strength = 0.7
        elif not negation_proof:
            # If no contradiction, assign base confidence
            proof_strength = 0.5
        
        return {
            "direct_proof": direct_proof,
            "negation_proof": negation_proof,
            "derived_proof": derived_proof,
            "proof_strength": proof_strength
        }

    def _can_derive(self, hypothesis: str, statements: List[str], context: Dict) -> bool:
        """Check if hypothesis can be derived from statements with one inference step"""
        # Try to apply all rules to see if we can get the hypothesis
        if self._apply_modus_ponens(statements, context) == [hypothesis]:
            return True
        if self._apply_modus_tollens(statements, context) == [hypothesis]:
            return True
        if self._apply_syllogism(statements, context) == [hypothesis]:
            return True
        if self._apply_disjunctive_syllogism(statements, context) == [hypothesis]:
            return True
        if self._apply_hypothetical_syllogism(statements, context) == [hypothesis]:
            return True
        return False

    def _check_contradictions(self, proof_steps: List[Dict], hypothesis: str, context: Dict) -> Dict[str, Any]:
        """Check for contradictions in the proof chain"""
        all_statements = []
        for step in proof_steps:
            all_statements.extend(step["input"])
            all_statements.append(step["output"])
        
        # Check for internal contradictions
        contradictions = self.identify_contradictions(all_statements)
        
        # Check if hypothesis contradicts existing statements
        hypothesis_contradictions = []
        for stmt in all_statements:
            if self._are_contradictory(stmt, hypothesis):
                hypothesis_contradictions.append(stmt)
        
        return {
            "internal_contradictions": contradictions,
            "hypothesis_contradictions": hypothesis_contradictions,
            "contradiction_score": len(contradictions) / len(all_statements) if all_statements else 0
        }

    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Check if two statements are contradictory"""
        negation_words = ["not", "false", "no", "never"]
        
        # Check direct negation
        for negation in negation_words:
            if statement1 == f"{negation} {statement2}" or statement2 == f"{negation} {statement1}":
                return True
        
        # Check semantic contradiction (simplified)
        if " is " in statement1 and " is " in statement2:
            subject1 = statement1.split(" is ")[0].strip()
            predicate1 = statement1.split(" is ")[1].strip()
            
            subject2 = statement2.split(" is ")[0].strip()
            predicate2 = statement2.split(" is ")[1].strip()
            
            if subject1 == subject2 and predicate1 != predicate2:
                return True
        
        return False

    def _determine_certainty(self, evaluation: Dict, contradiction_analysis: Dict) -> Tuple[bool, float]:
        """Determine final certainty of hypothesis"""
        # Start with proof strength
        certainty = evaluation["proof_strength"]
        
        # Adjust based on contradictions
        contradiction_score = contradiction_analysis["contradiction_score"]
        certainty -= contradiction_score * 0.5
        
        # Penalize if negation is proven
        if evaluation["negation_proof"]:
            certainty = min(certainty, 0.3)
        
        # Cap and determine proven status
        certainty = max(0.0, min(1.0, certainty))
        is_proven = certainty >= self.certainty_threshold and not contradiction_analysis["hypothesis_contradictions"]
        
        return is_proven, certainty

    def _format_result(self, is_proven: bool, certainty: float, proof_steps: List[Dict], 
                    premises: List[str], hypothesis: str, contradiction_analysis: Dict) -> Dict[str, Any]:
        """Format final result with proof structure"""
        return {
            "hypothesis": hypothesis,
            "proven": is_proven,
            "certainty": certainty,
            "premises": premises,
            "proof_steps": proof_steps,
            "contradictions": contradiction_analysis,
            "metrics": {
                "proof_length": len(proof_steps),
                "premises_used": len(premises),
                "contradictions_found": len(contradiction_analysis["internal_contradictions"]),
                "hypothesis_contradictions": len(contradiction_analysis["hypothesis_contradictions"]),
                "certainty_level": "high" if certainty > 0.8 else "medium" if certainty > 0.5 else "low"
            },
            "reasoning_type": "deductive"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Deductive ===\n")
    printer.status("TEST", "Starting Deductive Reasoning tests", "info")

    deduction = ReasoningDeductive()

    # Test 1: Simple modus ponens
    premises = [
        "If it rains, the ground is wet",
        "It is raining"
    ]
    hypothesis = "The ground is wet"
    
    result1 = deduction.perform_reasoning(
        premises=premises,
        hypothesis=hypothesis
    )
    printer.pretty("Test 1 Result", result1)

    # Test 2: Syllogism
    premises = [
        "All humans are mortal",
        "Socrates is a human"
    ]
    hypothesis = "Socrates is mortal"
    
    result2 = deduction.perform_reasoning(
        premises=premises,
        hypothesis=hypothesis
    )
    printer.pretty("Test 2 Result", result2)

    # Test 3: Contradiction detection
    premises = [
        "All birds can fly",
        "Penguins are birds",
        "Penguins cannot fly"
    ]
    hypothesis = "Penguins can fly"
    
    result3 = deduction.perform_reasoning(
        premises=premises,
        hypothesis=hypothesis
    )
    printer.pretty("Test 3 Result", result3)

    print("\n=== Successfully Ran Reasoning Deductive ===\n")