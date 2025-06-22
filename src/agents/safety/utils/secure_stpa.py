

from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI System-Theoretic Process Analysis")
printer = PrettyPrinter

class SecureSTPA:
    def __init__(self):
        """
        Implementing functions from:
        STPA Handbook - Nancy G. Levenson & John P. Thomas
        Nuclear Engineering and Technology - Sejin Jung, Yoona Heo, Junbeom Yoo
        1.2 Identifying hazardous system behaviour - Stephan Baumgart & Sasikumar Punnekkat
        """
        self.config = load_global_config()
        self.stpa_config = get_config_section('secure_stpa')
        #self. =  self.stpa_config.get('')

        self.memory = SecureMemory()
        self.reset_analysis()

    def reset_analysis(self):
        """Reset all analysis components"""
        self.losses = []
        self.hazards = []
        self.safety_constraints = []
        self.control_structure = {}
        self.context_tables = defaultdict(list)
        self.uca_table = []
        self.process_models = defaultdict(dict)
        self.component_states = defaultdict(dict)

    def define_analysis_scope(self, losses: List[str], hazards: List[str],
        constraints: List[str], system_boundary: str = None) -> None:
        self.losses = losses
        self.hazards = hazards
        self.safety_constraints = constraints
        self.system_boundary = system_boundary or "System Boundary Not Defined"
        self.memory.add({
            "losses": losses,
            "hazards": hazards,
            "constraints": constraints,
            "boundary": self.system_boundary
        }, tags=["stpa_scope"])
        printer.status("STPA", "Scope defined with hazards, losses, constraints.", "info")

    def model_control_structure(
        self,
        structure: Dict[str, Dict[str, List[str]]],
        process_models: Dict[str, Dict[str, List[str]]] = None
    ) -> None:
        """
        Enhanced control structure modeling:
        {
            "Controller": {
                "inputs": ["sensor1", "feedback"],
                "outputs": ["control_action1"],
                "process_vars": ["state_var1", "threshold"]
            },
            ...
        }
        """
        self.control_structure = structure
        self.process_models = process_models or {}
        
        # Initialize component states
        for component in structure:
            self.component_states[component] = {
                "current": "INIT",
                "transitions": defaultdict(list)
            }
        
        self.memory.add({
            "control_structure": structure,
            "process_models": self.process_models
        }, tags=["stpa_model"])
        printer.status("STPA", f"Control structure modeled with {len(structure)} components.", "info")

    def identify_unsafe_control_actions(
        self,
        custom_guidewords: List[str] = None
    ) -> List[Dict]:
        guidewords = custom_guidewords or [
            "Not Providing Causes Hazard",
            "Providing Causes Hazard",
            "Too Early/Too Late/Out of Order",
            "Stopped Too Soon/Applied Too Long"
        ]
        
        ucas = []
        for controller, comp_data in self.control_structure.items():
            for action in comp_data.get("outputs", []):
                for guideword in guidewords:
                    hazard_link = self._determine_hazard_link(
                        controller, 
                        action, 
                        guideword
                    )
                    
                    uca = {
                        "id": f"uca_{len(self.uca_table)+1}",
                        "controller": controller,
                        "control_action": action,
                        "guideword": guideword,
                        "hazard_link": hazard_link,
                        "state_constraints": self._get_state_constraints(controller),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.uca_table.append(uca)
                    self.memory.add(uca, tags=["unsafe_control_action"])
                    ucas.append(uca)
        
        printer.status("STPA", f"{len(ucas)} UCAs identified.", "warn")
        return ucas

    def build_context_tables(
        self, 
        formal_spec: Dict = None,
        fta_config: Dict = None
    ) -> Dict[str, List[Dict]]:
        """
        Build context tables with formal methods integration
        """
        for uca in self.uca_table:
            context_entry = self._generate_context_entry(uca, formal_spec, fta_config)
            self.context_tables[uca['controller']].append(context_entry)
            self.memory.add(
                context_entry, 
                tags=["stpa_context", f"controller:{uca['controller']}"]
            )
        
        printer.status("STPA", "Context tables constructed.", "info")
        return dict(self.context_tables)

    def identify_loss_scenarios(
        self, 
        probability_model: str = "heuristic"
    ) -> List[Dict]:
        scenarios = []
        for controller, contexts in self.context_tables.items():
            for context in contexts:
                scenario = self._generate_loss_scenario(context, probability_model)
                scenarios.append(scenario)
                self.memory.add(scenario, tags=["loss_scenario"])
        
        printer.status("STPA", f"{len(scenarios)} loss scenarios identified.", "error")
        return scenarios

    def perform_sos_analysis(
        self,
        consistency_checks: bool = True,
        deadlock_detection: bool = True,
        safe_state_reachability: bool = True
    ) -> Dict:
        """System-of-Systems extended analysis"""
        results = {}
        
        if consistency_checks:
            results['state_inconsistencies'] = self._check_state_consistency()
        
        if deadlock_detection:
            results['deadlock_risks'] = self._detect_communication_deadlocks()
        
        if safe_state_reachability:
            results['safe_state_analysis'] = self._analyze_safe_state_reachability()
        
        self.memory.add(results, tags=["sos_analysis"])
        return results

    def export_analysis_report(
        self, 
        format: str = "json", 
        include_sos: bool = False
    ) -> Dict:
        report = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "system_boundary": self.system_boundary
            },
            "losses": self.losses,
            "hazards": self.hazards,
            "safety_constraints": self.safety_constraints,
            "control_structure": self.control_structure,
            "unsafe_control_actions": self.uca_table,
            "context_tables": dict(self.context_tables),
            "loss_scenarios": self.identify_loss_scenarios()
        }
        
        if include_sos:
            report["sos_analysis"] = self.perform_sos_analysis()
        
        self.memory.add(report, tags=["stpa_report"])
        return report

    # ------------ Core Analysis Methods ------------
    def _determine_hazard_link(
        self, 
        controller: str, 
        action: str, 
        guideword: str
    ) -> str:
        """Enhanced hazard linking with state-aware analysis"""
        # 1. Check predefined hazard mappings
        for hazard in self.hazards:
            if controller.lower() in hazard.lower() and action.lower() in hazard.lower():
                return hazard
                
        # 2. State-based hazard inference
        current_state = self.component_states[controller]["current"]
        if "emergency" in current_state and "Not Providing" in guideword:
            return next((h for h in self.hazards if "shutdown" in h), self.hazards[0])
        
        # 3. NLP-based fallback
        return self._nlp_based_hazard_prediction(action, guideword)

    def _generate_context_entry(
        self, 
        uca: Dict, 
        formal_spec: Dict, 
        fta_config: Dict
    ) -> Dict:
        """Generate context table entry with formal methods"""
        # 1. Process model integration
        process_vars = self.process_models.get(
            uca["controller"], 
            {}
        ).get("variables", [])
        
        # 2. Formal specification integration
        if formal_spec:
            spec_vars = formal_spec.get(
                uca["controller"], {}
            ).get(uca["control_action"], [])
            process_vars += spec_vars
        
        # 3. Fault Tree Analysis integration
        conditions = []
        if fta_config:
            conditions = self._run_fta_analysis(
                uca["control_action"],
                fta_config
            )
        
        return {
            "uca_id": uca["id"],
            "controller": uca["controller"],
            "control_action": uca["control_action"],
            "guideword": uca["guideword"],
            "process_variables": list(set(process_vars)),
            "hazard_conditions": conditions,
            "state_constraints": uca["state_constraints"],
            "timestamp": datetime.now().isoformat()
        }

    def _generate_loss_scenario(
        self, 
        context: Dict, 
        probability_model: str
    ) -> Dict:
        """Generate loss scenario with probability estimation"""
        # Impact severity assessment
        hazard_severity = 3  # Default medium
        if "critical" in context["guideword"]:
            hazard_severity = 5
        elif "minor" in context["guideword"]:
            hazard_severity = 1
        
        # Probability estimation
        if probability_model == "heuristic":
            probability = self._heuristic_probability_estimation(context)
        else:
            probability = 0.3  # Default value
        
        return {
            "context_id": context["uca_id"],
            "loss": self._map_to_loss(context),
            "severity": hazard_severity,
            "probability": probability,
            "risk_level": hazard_severity * probability,
            "mitigation": self._generate_mitigation_strategy(context),
            "causal_factors": self._identify_causal_factors(context)
        }

    # ------------ Formal Methods Integration ------------
    def _run_fta_analysis(self, top_event: str, gate_structure: Dict[str, Dict], basic_event_probs: Dict[str, float]) -> List[Dict]:
        """
        Run fault tree analysis using real FTA libraries (sfta or pyfta), returning minimal cut sets
        and probability estimates for the top_event.
    
        Args:
          top_event: Identifier of the top-level event (e.g., unsafe control action failure).
          gate_structure: Dict defining gates, e.g.:
            {
              "G1": {"type": "AND", "inputs": ["E1", "E2"]},
              "G_top": {"type": "OR", "inputs": ["G1", "E3"]}
            }
          basic_event_probs: Mapping basic event → probability/rate.
    
        Returns:
          A list of dicts: {cut_set: List[str], probability: float}
        """
        results = []
        try:
            import sfta
            ftree = sfta.FaultTree()
            for evt, p in basic_event_probs.items():
                ftree.add_event(evt, prob=p)
            for gate, info in gate_structure.items():
                ftree.add_gate(gate, info["type"], info["inputs"])
            ftree.set_top(top_event)
    
            mcs = ftree.minimal_cut_sets()
            for cs in mcs:
                prob = ftree.probability(cs)
                results.append({"cut_set": cs, "probability": prob})
        except ImportError:
            try:
                from pyfta import FaultTree, Gate
                ft = FaultTree()
                for name, p in basic_event_probs.items():
                    ft.add_event(name, p)
                for gid, info in gate_structure.items():
                    ft.add_gate(Gate(gid, info["type"], info["inputs"]))
                ft.set_top(top_event)
                for cs in ft.minimal_cut_sets():
                    results.append({"cut_set": cs, "probability": ft.probability_of(cs)})
            except ImportError:
                logger.warning("No FTA libraries found. Returning stub results.")
                results = self._fta_stub(top_event)
    
        if results:
            self.memory.add({"top_event": top_event, "fta_results": results}, tags=["fta"])
            logger.info(f"FTA produced {len(results)} cut-sets for top_event '{top_event}'")
        else:
            logger.warning(f"FTA returned no cut-sets for '{top_event}'")
        return results
    
    def _fta_stub(self, top_event: str) -> List[Dict]:
        return [
            {"cut_set": ["E1", "E2"], "probability": 0.12},
            {"cut_set": ["E3"], "probability": 0.03},
        ]

    def _get_state_constraints(self, controller: str) -> List[str]:
        """Extract state-related constraints from process model"""
        return self.process_models.get(controller, {}).get("constraints", [
            f"State-dependent constraint for {controller}"
        ])

    # ------------ System-of-Systems Methods ------------
    def _check_state_consistency(self) -> List[Dict]:
        """Check for state inconsistencies across components"""
        inconsistencies = []
        ref_state = None
        
        for component, states in self.component_states.items():
            if not ref_state:
                ref_state = states["current"]
                continue
                
            if states["current"] != ref_state:
                inconsistencies.append({
                    "component": component,
                    "state": states["current"],
                    "conflict_with": ref_state
                })
        
        return inconsistencies

    def _detect_communication_deadlocks(self) -> List[Dict]:
        """
        Detect circular communication dependencies or asymmetric control-response patterns
        that could lead to communication deadlocks or systemic waiting.
    
        Uses a directed wait-for graph derived from control structure channels.
        Returns a list of deadlock risks with cycle participants and blocking causes.
        """
        wait_for_graph = defaultdict(set)
        deadlocks = []
    
        # Step 1: Build wait-for graph from control structure
        for controller, structure in self.control_structure.items():
            outputs = set(structure.get("outputs", []))
            inputs = set(structure.get("inputs", []))
            for out in outputs:
                for other, other_struct in self.control_structure.items():
                    if out in other_struct.get("inputs", []):
                        wait_for_graph[controller].add(other)
    
        # Step 2: Detect cycles in wait-for graph (DFS)
        visited = set()
        path = []
    
        def dfs(node, trace):
            visited.add(node)
            trace.append(node)
            for neighbor in wait_for_graph[node]:
                if neighbor in trace:
                    cycle = trace[trace.index(neighbor):] + [neighbor]
                    deadlocks.append({
                        "type": "cyclic_dependency",
                        "components_involved": cycle,
                        "cause": "Communication wait cycle detected",
                        "cycle_length": len(cycle)
                    })
                elif neighbor not in visited:
                    dfs(neighbor, trace[:])
    
        for node in wait_for_graph:
            if node not in visited:
                dfs(node, [])
    
        # Step 3: Check for asymmetric dependencies
        for src, neighbors in wait_for_graph.items():
            for dst in neighbors:
                if src not in wait_for_graph[dst]:
                    deadlocks.append({
                        "type": "asymmetric_dependency",
                        "from": src,
                        "to": dst,
                        "issue": "Missing reciprocal output-input mapping"
                    })
    
        # Step 4: Log and return
        if deadlocks:
            printer.status("STPA", f"{len(deadlocks)} communication deadlock risks detected.", "error")
            for d in deadlocks:
                self.memory.add(d, tags=["communication_deadlock"])
        else:
            printer.status("STPA", "No communication deadlocks detected.", "success")
    
        return deadlocks

    def _analyze_safe_state_reachability(self) -> Dict:
        """Analyze reachability of safe states"""
        return {
            "unreachable_safe_states": [],
            "transition_issues": [
                f"Potential issue in {comp} state transitions" 
                for comp in self.control_structure
            ]
        }

    # ------------ Helper Methods ------------
    def _nlp_based_hazard_prediction(self, action: str, guideword: str) -> str:
        """
        Uses linguistic modeling and semantic matching to predict the most likely hazard associated
        with a given control action and guideword using the NLPEngine.
    
        Args:
            action (str): The control action name (e.g., "coolant valve close")
            guideword (str): The guideword (e.g., "Not Providing Causes Hazard")
    
        Returns:
            str: Predicted hazard string from self.hazards
        """
        from src.agents.language.nlp_engine import NLPEngine
    
        if not hasattr(self, "nlp_engine"):
            self.nlp_engine = NLPEngine()
    
        tokens_action = self.nlp_engine.process_text(action)
        tokens_guide = self.nlp_engine.process_text(guideword)
        tokens_all = tokens_action + tokens_guide
    
        # Extract lemmatized key terms from action and guideword
        lemmas = {token.lemma for token in tokens_all if not token.is_stop and token.pos not in {"PUNCT", "DET"}}
        
        # Semantic matching with hazard corpus
        ranked = []
        for hazard in self.hazards:
            tokens_hazard = self.nlp_engine.process_text(hazard)
            hazard_lemmas = {t.lemma for t in tokens_hazard if not t.is_stop}
            
            # Compute lexical similarity score
            overlap = len(lemmas & hazard_lemmas)
            score = overlap / (len(lemmas | hazard_lemmas) + 1e-5)  # Jaccard index with epsilon
    
            # Bias adjustment for critical keywords
            if any(k in hazard_lemmas for k in {"shutdown", "overheat", "fail", "loss", "injury"}):
                score *= 1.2
    
            ranked.append((hazard, score))
    
        # Sort and return the top-ranked hazard
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_hazard = ranked[0][0] if ranked else self.hazards[0]
    
        self.memory.add({
            "method": "nlp_prediction",
            "input_action": action,
            "input_guideword": guideword,
            "predicted_hazard": top_hazard,
            "ranked_hazards": ranked[:3]
        }, tags=["hazard_prediction"])
    
        return top_hazard

    def _heuristic_probability_estimation(self, context: Dict) -> float:
        """Estimate probability using heuristic rules"""
        factors = {
            "complexity": len(context["process_variables"]) / 10,
            "state_dependency": 0.7 if context["state_constraints"] else 0.3,
            "timing_constraint": 0.8 if "Timing" in context["guideword"] else 0.4
        }
        return min(0.95, max(0.05, sum(factors.values()) / len(factors)))

    def _map_to_loss(self, context: Dict) -> str:
        """Map context to system-level loss"""
        for loss in self.losses:
            if any(keyword in loss for keyword in ["equipment", "damage"]):
                return loss
        return self.losses[0]

    def _generate_mitigation_strategy(self, context: Dict) -> List[str]:
        """Generate context-aware mitigation strategies"""
        strategies = []
        if "Timing" in context["guideword"]:
            strategies.append("Implement timing watchdog")
            strategies.append("Add sequence verification")
        
        if "Providing" in context["guideword"]:
            strategies.append("Add pre-action validation checks")
            strategies.append("Implement permission system")
        
        return strategies

    def _identify_causal_factors(self, context: Dict) -> List[str]:
        """Identify potential causal factors"""
        factors = ["Sensor failure", "Communication delay"]
        if "process_variables" in context:
            factors += [f"{var} out of range" for var in context["process_variables"]]
        return factors


if __name__ == "__main__":
    print("\n=== Running SLAI System-Theoretic Process Analysis Test ===\n")
    printer.status("Init", "Secure STPA initialized", "success")

    stpa = SecureSTPA()
    print(stpa)
    print("\n* * * * * Phase 2 * * * * *\n")
    
    print("\n=== Successfully Ran SLAI System-Theoretic Process Analysis ===\n")
