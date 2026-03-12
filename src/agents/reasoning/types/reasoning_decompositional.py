
from typing import Any, Dict, List, Optional, Tuple, Set

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Decompositional")
printer = PrettyPrinter

class ReasoningDecompositional(BaseReasoning):
    """
    Implements decompositional reasoning: Breaking down systems into components
    and analyzing their structure, function, and interactions
    Process:
    1. Decompose system into constituent parts
    2. Analyze individual components
    3. Identify component relationships and interactions
    4. Synthesize understanding of the whole system
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.decompositional_config = get_config_section("reasoning_decompositional")
        self.max_depth = self.decompositional_config.get('max_depth')
        self.min_component_size = self.decompositional_config.get('min_component_size')
        self.structural_weight = self.decompositional_config.get('structural_weight')
        self.functional_weight = self.decompositional_config.get('functional_weight')
        self.enable_interaction_analysis = self.decompositional_config.get('enable_interaction_analysis')

    def perform_reasoning(self, system: Any, context: dict = None) -> Dict[str, Any]:
        """
        Perform decompositional reasoning on a system
        Args:
            system: The entity or system to decompose
            context: Additional context for decomposition
        Returns:
            Hierarchical decomposition with analysis
        """
        self.log_step("Starting decompositional reasoning")
        context = context or {}
        
        # Step 1: Decompose system
        decomposition_tree = self.decompose_system(system, context)
        
        # Step 2: Analyze components
        component_analysis = self.analyze_components(decomposition_tree, context)
        
        # Step 3: Identify interactions
        interaction_analysis = {}
        if self.enable_interaction_analysis:
            interaction_analysis = self.analyze_interactions(decomposition_tree, context)
        
        # Step 4: Synthesize understanding
        system_understanding = self.synthesize_understanding(
            decomposition_tree, 
            component_analysis, 
            interaction_analysis, 
            context
        )
        
        # Format and return results
        return self._format_results(
            decomposition_tree, 
            component_analysis, 
            interaction_analysis, 
            system_understanding
        )

    def decompose_system(self, system: Any, context: Dict, depth: int = 0, parent_id: str = None) -> Dict:
        """
        Recursively decompose a system into hierarchical components
        Args:
            system: The system to decompose
            context: Decomposition context
            depth: Current recursion depth
            parent_id: ID of parent component
        Returns:
            Hierarchical decomposition tree
        """
        # Base case: stop recursion at max depth or min size
        if depth >= self.max_depth:
            return {
                "id": self._generate_component_id(system, parent_id),
                "name": self._get_component_name(system),
                "type": "atomic",
                "depth": depth,
                "content": system,
                "children": []
            }
        
        # Decompose using appropriate strategy
        decomposition_strategy = context.get("decomposition_strategy", "functional")
        components = self.decompose_whole(system, decomposition_strategy)
        
        # Create component node
        component_id = self._generate_component_id(system, parent_id)
        node = {
            "id": component_id,
            "name": self._get_component_name(system),
            "type": "composite",
            "depth": depth,
            "content": system,
            "children": []
        }
        
        # Recursively decompose children
        for i, component in enumerate(components):
            if self._is_decomposable(component, depth):
                child = self.decompose_system(
                    component, 
                    context, 
                    depth + 1,
                    parent_id=f"{component_id}-{i}"
                )
                node["children"].append(child)
            else:
                node["children"].append({
                    "id": f"{component_id}-{i}",
                    "name": self._get_component_name(component),
                    "type": "atomic",
                    "depth": depth + 1,
                    "content": component,
                    "children": []
                })
        
        return node

    def _is_decomposable(self, component: Any, depth: int) -> bool:
        """Determine if a component should be further decomposed"""
        # Check size constraints
        if isinstance(component, (list, dict, set)):
            size = len(component)
        elif hasattr(component, "__len__"):
            size = len(component)
        else:
            size = 1
            
        return size > self.min_component_size and depth < self.max_depth

    def _generate_component_id(self, component: Any, parent_id: str) -> str:
        """Generate unique component ID"""
        base_name = self._get_component_name(component).replace(" ", "_").lower()
        if parent_id:
            return f"{parent_id}_{base_name}"
        return f"root_{base_name}"

    def _get_component_name(self, component: Any) -> str:
        """Extract component name"""
        if isinstance(component, dict):
            return component.get("name", str(component))
        elif hasattr(component, "name"):
            return component.name
        return str(component)

    def analyze_components(self, decomposition_tree: Dict, context: Dict) -> Dict[str, Dict]:
        """
        Analyze individual components in the decomposition
        Args:
            decomposition_tree: Hierarchical decomposition
            context: Analysis context
        Returns:
            Dictionary of component analyses keyed by component ID
        """
        analysis = {}
        
        # BFS traversal of decomposition tree
        queue = [decomposition_tree]
        while queue:
            node = queue.pop(0)
            
            # Analyze current node
            analysis[node["id"]] = self._analyze_component(node, context)
            
            # Add children to queue
            queue.extend(node["children"])
        
        return analysis

    def _analyze_component(self, component_node: Dict, context: Dict) -> Dict[str, Any]:
        """Perform in-depth analysis of a single component"""
        component = component_node["content"]
        
        # Structural analysis
        structure = self._analyze_structure(component)
        
        # Functional analysis
        function = self._analyze_function(component, context)
        
        # Complexity metrics
        complexity = self._calculate_complexity(component_node)
        
        return {
            "structure": structure,
            "function": function,
            "complexity": complexity,
            "criticality": self._assess_criticality(component_node, context),
            "dependencies": self._identify_dependencies(component, context)
        }

    def _analyze_structure(self, component: Any) -> Dict:
        """Analyze structural properties of component"""
        if isinstance(component, dict):
            return {
                "type": "dictionary",
                "key_count": len(component),
                "keys": list(component.keys()),
                "value_types": {k: type(v).__name__ for k, v in component.items()}
            }
        elif isinstance(component, list):
            return {
                "type": "list",
                "length": len(component),
                "element_types": [type(item).__name__ for item in component],
                "unique_types": len(set(type(item).__name__ for item in component))
            }
        elif isinstance(component, set):
            return {
                "type": "set",
                "size": len(component),
                "element_types": [type(item).__name__ for item in component],
                "unique_types": len(set(type(item).__name__ for item in component))
            }
        else:
            return {
                "type": type(component).__name__,
                "attributes": dir(component) if hasattr(component, "__dict__") else [],
                "methods": [m for m in dir(component) if callable(getattr(component, m))] 
                           if hasattr(component, "__dict__") else []
            }

    def _analyze_function(self, component: Any, context: Dict) -> str:
        """Determine the function/purpose of a component"""
        # Use context-specific function mapping if available
        if "function_mapping" in context:
            for pattern, function in context["function_mapping"].items():
                if pattern in str(component):
                    return function
        
        # Default behavior based on component type
        if isinstance(component, dict):
            return "Data storage and key-value mapping"
        elif isinstance(component, list):
            return "Sequential data storage and processing"
        elif isinstance(component, set):
            return "Unique element storage and set operations"
        elif callable(component):
            return "Functional operation"
        return "Data representation"

    def _calculate_complexity(self, component_node: Dict) -> float:
        """Calculate complexity metric for component"""
        # Simple complexity: based on children count and depth
        child_count = len(component_node["children"])
        depth_factor = component_node["depth"] / self.max_depth
        return min(1.0, 0.3 * child_count + 0.7 * depth_factor)

    def _assess_criticality(self, component_node: Dict, context: Dict) -> float:
        """Assess how critical the component is to system operation"""
        # Root is always critical
        if component_node["depth"] == 0:
            return 1.0
        
        # Check context-specific critical components
        critical_components = context.get("critical_components", [])
        if component_node["name"] in critical_components:
            return 0.9
        
        # Components with many children are more critical
        child_count = len(component_node["children"])
        return min(0.8, 0.1 * child_count)

    def _identify_dependencies(self, component: Any, context: Dict) -> List[str]:
        """Identify dependencies of the component"""
        dependencies = []
        
        if isinstance(component, dict):
            # Dependencies are keys that reference other components
            for value in component.values():
                if isinstance(value, str) and value.startswith("comp_"):
                    dependencies.append(value)
        elif hasattr(component, "dependencies"):
            dependencies = list(component.dependencies)
        
        return dependencies

    def analyze_interactions(self, decomposition_tree: Dict, context: Dict) -> Dict[str, List[Dict]]:
        """
        Analyze interactions between components
        Args:
            decomposition_tree: Hierarchical decomposition
            context: Analysis context
        Returns:
            Dictionary of interaction analyses keyed by interaction type
        """
        interactions = {
            "structural": [],
            "functional": [],
            "data_flow": [],
            "control_flow": []
        }
        
        # Get all components
        components = self._flatten_decomposition(decomposition_tree)
        
        # Analyze pairwise interactions
        component_ids = list(components.keys())
        for i in range(len(component_ids)):
            for j in range(i + 1, len(component_ids)):
                comp1 = components[component_ids[i]]
                comp2 = components[component_ids[j]]
                
                # Only analyze interactions at the same level
                if comp1["depth"] == comp2["depth"]:
                    interaction = self._analyze_interaction(comp1, comp2, context)
                    if interaction:
                        interactions[interaction["type"]].append(interaction)
        
        # Analyze hierarchical interactions
        interactions["hierarchical"] = self._analyze_hierarchical_interactions(
            decomposition_tree, 
            context
        )
        
        return interactions

    def _flatten_decomposition(self, node: Dict) -> Dict[str, Dict]:
        """Flatten hierarchical decomposition to dictionary"""
        components = {node["id"]: node}
        for child in node["children"]:
            components.update(self._flatten_decomposition(child))
        return components

    def _analyze_interaction(self, comp1: Dict, comp2: Dict, context: Dict) -> Optional[Dict]:
        """Analyze interaction between two components"""
        # Check for structural connections
        structural_score = self._calculate_structural_similarity(comp1, comp2)
        
        # Check functional dependencies
        functional_score = self._calculate_functional_dependency(comp1, comp2, context)
        
        # Determine interaction type based on scores
        if structural_score > 0.7 and functional_score > 0.7:
            return {
                "type": "structural_functional",
                "components": [comp1["id"], comp2["id"]],
                "structural_score": structural_score,
                "functional_score": functional_score,
                "description": f"Strong structural and functional relationship between {comp1['name']} and {comp2['name']}"
            }
        elif structural_score > 0.7:
            return {
                "type": "structural",
                "components": [comp1["id"], comp2["id"]],
                "structural_score": structural_score,
                "description": f"Structural relationship between {comp1['name']} and {comp2['name']}"
            }
        elif functional_score > 0.7:
            return {
                "type": "functional",
                "components": [comp1["id"], comp2["id"]],
                "functional_score": functional_score,
                "description": f"Functional dependency between {comp1['name']} and {comp2['name']}"
            }
        return None

    def _calculate_structural_similarity(
        self, 
        comp1: Dict, 
        comp2: Dict
    ) -> float:
        """Calculate structural similarity between components"""
        struct1 = comp1.get("analysis", {}).get("structure", {})
        struct2 = comp2.get("analysis", {}).get("structure", {})
        
        # Simple similarity based on structure type
        if struct1.get("type") == struct2.get("type"):
            return 0.8
        return 0.2

    def _calculate_functional_dependency(self, comp1: Dict, comp2: Dict, context: Dict) -> float:
        """Calculate functional dependency between components"""
        func1 = comp1.get("analysis", {}).get("function", "")
        func2 = comp2.get("analysis", {}).get("function", "")
        
        # Check if functions are complementary
        complementary_pairs = context.get("complementary_functions", [])
        for pair in complementary_pairs:
            if func1 in pair and func2 in pair:
                return 0.9
        
        # Check if one component's dependencies include the other
        if comp2["id"] in comp1.get("analysis", {}).get("dependencies", []):
            return 0.7
        if comp1["id"] in comp2.get("analysis", {}).get("dependencies", []):
            return 0.7
            
        return 0.0

    def _analyze_hierarchical_interactions(self, node: Dict, context: Dict) -> List[Dict]:
        """Analyze interactions within hierarchical relationships"""
        interactions = []
        
        # Parent-child interactions
        for child in node["children"]:
            interaction = {
                "type": "parent_child",
                "parent": node["id"],
                "child": child["id"],
                "direction": "parent_to_child",
                "description": f"{node['name']} contains {child['name']}"
            }
            interactions.append(interaction)
            
            # Recursively analyze child's hierarchy
            interactions.extend(self._analyze_hierarchical_interactions(child, context))
        
        # Sibling interactions
        if len(node["children"]) > 1:
            for i in range(len(node["children"])):
                for j in range(i + 1, len(node["children"])):
                    comp1 = node["children"][i]
                    comp2 = node["children"][j]
                    sibling_interaction = self._analyze_interaction(comp1, comp2, context)
                    if sibling_interaction:
                        sibling_interaction["type"] = "sibling_" + sibling_interaction["type"]
                        interactions.append(sibling_interaction)
        
        return interactions

    def synthesize_understanding(self, decomposition_tree: Dict, component_analysis: Dict,
                                interaction_analysis: Dict, context: Dict) -> Dict[str, Any]:
        """
        Synthesize understanding of the whole system from components and interactions
        Args:
            decomposition_tree: Hierarchical decomposition
            component_analysis: Analysis of individual components
            interaction_analysis: Analysis of component interactions
            context: Synthesis context
        Returns:
            Comprehensive system understanding
        """
        # Calculate system metrics
        all_components = self._flatten_decomposition(decomposition_tree)
        component_count = len(all_components)
        max_depth = max(comp["depth"] for comp in all_components.values())
        avg_complexity = sum(
            analysis.get("complexity", 0) 
            for analysis in component_analysis.values()
        ) / component_count if component_count else 0
        
        # Identify key components
        key_components = [
            comp_id for comp_id, analysis in component_analysis.items() 
            if analysis.get("criticality", 0) > 0.7
        ]
        
        # Determine system properties
        system_properties = {
            "modularity": self._calculate_modularity(interaction_analysis),
            "cohesion": self._calculate_cohesion(component_analysis, interaction_analysis),
            "coupling": self._calculate_coupling(interaction_analysis)
        }
        
        # Generate insights
        insights = self._generate_insights(
            decomposition_tree, 
            component_analysis, 
            interaction_analysis,
            context
        )
        
        return {
            "system_properties": system_properties,
            "key_components": key_components,
            "system_metrics": {
                "component_count": component_count,
                "max_depth": max_depth,
                "avg_complexity": avg_complexity,
                "interaction_density": len(interaction_analysis.get("structural", [])) / component_count 
                                       if component_count else 0
            },
            "insights": insights,
            "vulnerabilities": self._identify_vulnerabilities(
                component_analysis, 
                interaction_analysis,
                context
            )
        }

    def _calculate_modularity(self, interactions: Dict) -> float:
        """Calculate system modularity score"""
        hierarchical = len(interactions.get("hierarchical", []))
        functional = len(interactions.get("functional", []))
        structural = len(interactions.get("structural", []))
        total_interactions = hierarchical + functional + structural
        
        # Higher modularity when hierarchical interactions dominate
        if total_interactions > 0:
            return hierarchical / total_interactions
        return 1.0  # No interactions = perfectly modular

    def _calculate_cohesion(self, component_analysis: Dict, interactions: Dict) -> float:
        """Calculate system cohesion score"""
        functional_groups = {}
        
        # Group components by function
        for comp_id, analysis in component_analysis.items():
            function = analysis.get("function", "unknown")
            if function not in functional_groups:
                functional_groups[function] = []
            functional_groups[function].append(comp_id)
        
        # Calculate intra-group interaction density
        intra_group_interactions = 0
        for group, components in functional_groups.items():
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    # Check if interaction exists between these components
                    for int_type in ["functional", "structural"]:
                        for interaction in interactions.get(int_type, []):
                            if (components[i] in interaction["components"] and 
                                components[j] in interaction["components"]):
                                intra_group_interactions += 1
                                break
        
        # Calculate possible intra-group interactions
        possible_interactions = sum(
            len(group) * (len(group) - 1) / 2 
            for group in functional_groups.values()
        )
        
        if possible_interactions > 0:
            return intra_group_interactions / possible_interactions
        return 0.0

    def _calculate_coupling(self, interactions: Dict) -> float:
        """Calculate system coupling score"""
        # Count inter-component interactions
        inter_component_interactions = sum(
            len(interactions[int_type]) 
            for int_type in ["functional", "structural", "data_flow", "control_flow"]
        )
        
        # Count hierarchical interactions (considered lower coupling)
        hierarchical_interactions = len(interactions.get("hierarchical", []))
        
        total_interactions = inter_component_interactions + hierarchical_interactions
        
        if total_interactions > 0:
            return inter_component_interactions / total_interactions
        return 0.0

    def _generate_insights(self, decomposition_tree: Dict, component_analysis: Dict,
                            interaction_analysis: Dict, context: Dict) -> List[str]:
        """Generate insights about system structure and behavior"""
        insights = []
        
        # Insight 1: Identify complex components
        complex_components = [
            comp_id for comp_id, analysis in component_analysis.items()
            if analysis.get("complexity", 0) > 0.8
        ]
        if complex_components:
            insights.append(
                f"High complexity components detected: {', '.join(complex_components)}. "
                "Consider refactoring for maintainability."
            )
        
        # Insight 2: Identify critical paths
        critical_paths = self._find_critical_paths(decomposition_tree, component_analysis)
        if critical_paths:
            insights.append(
                f"Critical paths identified: {len(critical_paths)} paths with high criticality components. "
                "These represent potential single points of failure."
            )
        
        # Insight 3: Coupling vs cohesion analysis
        if self._calculate_cohesion(component_analysis, interaction_analysis) < 0.3:
            insights.append(
                "Low system cohesion detected. Components show weak functional grouping. "
                "Consider reorganizing components to strengthen functional relationships."
            )
        
        return insights

    def _find_critical_paths(self, node: Dict, component_analysis: Dict) -> List[List[str]]:
        """Find paths with high criticality components"""
        paths = []
        
        # Leaf node
        if not node["children"]:
            if component_analysis.get(node["id"], {}).get("criticality", 0) > 0.7:
                return [[node["id"]]]
            return []
        
        # Process children
        for child in node["children"]:
            child_paths = self._find_critical_paths(child, component_analysis)
            for path in child_paths:
                # Add current node if critical or path contains critical components
                if (component_analysis.get(node["id"], {}).get("criticality", 0) > 0.7 or 
                    any(component_analysis.get(p, {}).get("criticality", 0) > 0.7 for p in path)):
                    paths.append([node["id"]] + path)
        
        return paths

    def _identify_vulnerabilities(self, component_analysis: Dict,
                                interaction_analysis: Dict, context: Dict) -> List[Dict]:
        """Identify system vulnerabilities based on analysis"""
        vulnerabilities = []
        
        # Vulnerability 1: Highly critical components without redundancy
        for comp_id, analysis in component_analysis.items():
            if (analysis.get("criticality", 0) > 0.8 and 
                not self._has_redundancy(comp_id, interaction_analysis)):
                vulnerabilities.append({
                    "component": comp_id,
                    "type": "single_point_of_failure",
                    "severity": "high",
                    "description": f"Critical component {comp_id} has no redundancy"
                })
        
        # Vulnerability 2: Highly coupled components
        for interaction in interaction_analysis.get("structural", []):
            if len(interaction["components"]) == 2:
                comp1, comp2 = interaction["components"]
                if (component_analysis[comp1].get("criticality", 0) > 0.7 and 
                    component_analysis[comp2].get("criticality", 0) > 0.7):
                    vulnerabilities.append({
                        "components": [comp1, comp2],
                        "type": "critical_coupling",
                        "severity": "medium",
                        "description": f"Critical components {comp1} and {comp2} are tightly coupled"
                    })
        
        return vulnerabilities

    def _has_redundancy(self, comp_id: str, interactions: Dict) -> bool:
        """Check if a component has redundancy"""
        # Look for similar components that could serve as backups
        for interaction in interactions.get("structural", []):
            if comp_id in interaction["components"]:
                other_comp = next(c for c in interaction["components"] if c != comp_id)
                if interaction.get("structural_score", 0) > 0.8:
                    return True
        return False

    def _format_results(self, decomposition_tree: Dict, component_analysis: Dict,
                        interaction_analysis: Dict, system_understanding: Dict ) -> Dict[str, Any]:
        """Format final results with metadata"""
        all_components = self._flatten_decomposition(decomposition_tree)
        return {
            "decomposition_tree": decomposition_tree,
            "component_analysis": component_analysis,
            "interaction_analysis": interaction_analysis,
            "system_understanding": system_understanding,
            "metrics": {
                "total_components": len(all_components),
                "max_depth": max(comp["depth"] for comp in all_components.values()) 
                             if all_components else 0,
                "interaction_count": sum(len(ia) for ia in interaction_analysis.values()),
                "critical_components": len(system_understanding.get("key_components", [])),
                "vulnerabilities": len(system_understanding.get("vulnerabilities", []))
            },
            "reasoning_type": "decompositional"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Decompositional ===\n")
    printer.status("TEST", "Starting Decompositional Reasoning tests", "info")

    decomposer = ReasoningDecompositional()

    # Test system: E-commerce platform
    ecommerce_system = {
        "name": "E-commerce Platform",
        "components": {
            "user_management": {
                "name": "User Management",
                "functions": ["authentication", "profile_management"],
                "dependencies": ["database"]
            },
            "product_catalog": {
                "name": "Product Catalog",
                "functions": ["product_search", "inventory_management"],
                "dependencies": ["database"]
            },
            "order_processing": {
                "name": "Order Processing",
                "functions": ["checkout", "payment_processing"],
                "dependencies": ["user_management", "product_catalog", "payment_gateway"]
            },
            "payment_gateway": {
                "name": "Payment Gateway",
                "functions": ["payment_authorization", "transaction_processing"],
                "dependencies": ["external_api"]
            },
            "database": {
                "name": "Database",
                "functions": ["data_storage", "data_retrieval"]
            },
            "external_api": {
                "name": "External APIs",
                "functions": ["third_party_integrations"]
            }
        }
    }

    context = {
        "critical_components": ["Database", "Payment Gateway"],
        "decomposition_strategy": "functional",
        "function_mapping": {
            "database": "Data persistence",
            "gateway": "Financial transactions"
        }
    }

    result = decomposer.perform_reasoning(
        system=ecommerce_system,
        context=context
    )
    
    printer.pretty("Decompositional Analysis Result", result)
    print("\n=== Successfully Ran Reasoning Decompositional ===\n")