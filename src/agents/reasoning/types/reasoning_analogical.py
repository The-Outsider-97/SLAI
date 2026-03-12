
from typing import Any, Dict, List, Optional, Tuple

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Analogical")
printer = PrettyPrinter

class ReasoningAnalogical(BaseReasoning):
    """
    Implements analogical reasoning: Finding similarities between domains and transferring knowledge
    Process:
    1. Identify structural and functional similarities
    2. Map relationships between source and target domains
    3. Transfer knowledge based on mappings
    4. Validate and adapt transferred knowledge
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.analogical_config = get_config_section("reasoning_analogical")
        self.min_similarity = self.analogical_config.get('min_similarity')
        self.max_analogies = self.analogical_config.get('max_analogies')
        self.structural_weight = self.analogical_config.get('structural_weight')
        self.functional_weight = self.analogical_config.get('functional_weight')
        self.adaptation_threshold = self.analogical_config.get('adaptation_threshold')

    def perform_reasoning(self, target: Any, source_domain: List[Any], context: dict = None) -> Dict[str, Any]:
        """
        Perform analogical reasoning between target and source domains
        Args:
            target: The subject to understand
            source_domain: Collection of potential analogs
            context: Additional context for reasoning
        Returns:
            Analogical mapping with transferred knowledge
        """
        self.log_step("Starting analogical reasoning process")
        context = context or {}
        
        # Step 1: Find suitable analogies
        analogies = self.find_analogies(target, source_domain)
        filtered_analogies = self._filter_analogies(analogies)
        
        if not filtered_analogies:
            self.log_step("No suitable analogies found", "warning")
            return self._format_results({}, analogies, context)
        
        # Step 2: Create mappings
        mappings = []
        for analogy in filtered_analogies:
            mapping = self.create_mapping(target, analogy["item"], context)
            mappings.append(mapping)
        
        # Step 3: Transfer knowledge
        transferred_knowledge = []
        for mapping in mappings:
            knowledge = self.transfer_knowledge(mapping, context)
            if knowledge:
                transferred_knowledge.append(knowledge)
        
        # Step 4: Select best transfer
        best_transfer = self.select_best_transfer(transferred_knowledge)
        
        # Format and return results
        return self._format_results(best_transfer, analogies, context)

    def _filter_analogies(self, analogies: List[Dict]) -> List[Dict]:
        """Filter analogies based on similarity threshold"""
        return [
            a for a in analogies 
            if a["similarity"] >= self.min_similarity
        ][:self.max_analogies]

    def create_mapping(self, target: Any, source: Any, context: Dict) -> Dict[str, Any]:
        """
        Create structural and functional mapping between target and source
        Args:
            target: The subject to understand
            source: Analogous item from source domain
            context: Additional context
        Returns:
            Mapping dictionary with correspondence details
        """
        self.log_step(f"Creating mapping between target and {source.get('name', 'source')}")
        
        # Extract properties
        target_props = self._extract_properties(target)
        source_props = self._extract_properties(source)
        
        # Identify correspondences
        correspondences = []
        for t_prop in target_props:
            for s_prop in source_props:
                if self._is_corresponding(t_prop, s_prop, context):
                    correspondences.append({
                        "target_property": t_prop,
                        "source_property": s_prop,
                        "similarity": self._property_similarity(t_prop, s_prop)
                    })
        
        # Calculate mapping quality
        structural_score = self._calculate_structural_score(target_props, source_props, correspondences)
        functional_score = self._calculate_functional_score(target, source, context)
        
        mapping_score = (
            self.structural_weight * structural_score +
            self.functional_weight * functional_score
        )
        
        return {
            "target": target,
            "source": source,
            "correspondences": correspondences,
            "structural_score": structural_score,
            "functional_score": functional_score,
            "mapping_score": mapping_score
        }

    def _is_corresponding(self, target_prop: str, source_prop: str, context: Dict) -> bool:
        """Determine if properties correspond based on similarity and context"""
        # Basic similarity check
        similarity = self._property_similarity(target_prop, source_prop)
        if similarity < self.min_similarity:
            return False
        
        # Contextual constraints
        constraints = context.get("property_constraints", {})
        if target_prop in constraints and source_prop not in constraints[target_prop]:
            return False
            
        return True

    def _property_similarity(self, prop1: str, prop2: str) -> float:
        """Calculate semantic similarity between properties"""
        # Simple implementation - would use word embeddings in real system
        words1 = set(prop1.lower().split("_"))
        words2 = set(prop2.lower().split("_"))
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0

    def _calculate_structural_score(self,target_props: set,source_props: set,correspondences: List[Dict]) -> float:
        """Calculate structural similarity score"""
        mapped_target = {c["target_property"] for c in correspondences}
        mapped_source = {c["source_property"] for c in correspondences}
        
        target_coverage = len(mapped_target) / len(target_props) if target_props else 0
        source_coverage = len(mapped_source) / len(source_props) if source_props else 0
        
        return (target_coverage + source_coverage) / 2

    def _calculate_functional_score(self, target: Any, source: Any, context: Dict) -> float:
        """Calculate functional similarity based on behaviors and relationships"""
        # Placeholder - would use domain-specific functional analysis
        target_funcs = self._extract_functions(target)
        source_funcs = self._extract_functions(source)
        
        if not target_funcs or not source_funcs:
            return 0.5  # Neutral score if no functional data
        
        similarity = self._calculate_similarity(
            set(target_funcs), 
            set(source_funcs)
        )
        return similarity

    def _extract_functions(self, entity: Any) -> List[str]:
        """Extract functional properties of an entity"""
        if isinstance(entity, dict):
            return entity.get("functions", []) + entity.get("capabilities", [])
        elif hasattr(entity, 'functions'):
            return entity.functions
        return []

    def transfer_knowledge(self, mapping: Dict, context: Dict) -> Dict[str, Any]:
        """
        Transfer knowledge from source to target based on mapping
        Args:
            mapping: Created mapping dictionary
            context: Additional context
        Returns:
            Transferred knowledge with adaptation details
        """
        target = mapping["target"]
        source = mapping["source"]
        self.log_step(f"Transferring knowledge from {source.get('name', 'source')}")
        
        # Identify transferable attributes
        transferable = []
        for corr in mapping["correspondences"]:
            source_prop = corr["source_property"]
            target_prop = corr["target_property"]
            
            if hasattr(source, source_prop) or isinstance(source, dict) and source_prop in source:
                transferable.append({
                    "source_property": source_prop,
                    "target_property": target_prop,
                    "value": source[source_prop] if isinstance(source, dict) else getattr(source, source_prop)
                })
        
        # Apply adaptations based on context
        adapted_knowledge = self.adapt_knowledge(transferable, target, context)
        
        return {
            "source": source,
            "original_knowledge": transferable,
            "adapted_knowledge": adapted_knowledge,
            "mapping_score": mapping["mapping_score"]
        }

    def adapt_knowledge(self, knowledge: List[Dict], target: Any, context: Dict) -> List[Dict]:
        """
        Adapt transferred knowledge to fit target domain
        Args:
            knowledge: Knowledge to adapt
            target: Target domain
            context: Adaptation context
        Returns:
            Adapted knowledge items
        """
        adapted = []
        constraints = context.get("constraints", {})
        
        for item in knowledge:
            # Check if direct transfer is possible
            if self._is_directly_applicable(item, target, constraints):
                adapted.append(item)
                continue
                
            # Apply transformations
            transformed = self._apply_transformations(item, target, constraints)
            if transformed:
                adapted.append(transformed)
        
        return adapted

    def _is_directly_applicable(self, item: Dict, target: Any, constraints: Dict) -> bool:
        """Check if knowledge can be directly applied to target"""
        prop = item["target_property"]
        
        # Check if property exists in target
        if isinstance(target, dict) and prop in target:
            return False  # Already exists, no need to transfer
        if hasattr(target, prop):
            return False
            
        # Check constraints
        if prop in constraints and "no_transfer" in constraints[prop]:
            return False
            
        return True

    def _apply_transformations(self, item: Dict, target: Any, constraints: Dict) -> Optional[Dict]:
        """Apply domain-specific transformations to knowledge"""
        prop = item["target_property"]
        value = item["value"]
        
        # Simple scaling transformation
        if isinstance(value, (int, float)):
            scale_factor = constraints.get(prop, {}).get("scale_factor", 1.0)
            new_value = value * scale_factor
            return {
                **item,
                "value": new_value,
                "transformation": f"scaled by {scale_factor}"
            }
        
        # Conceptual transformation
        concept_map = constraints.get("concept_mappings", {})
        if isinstance(value, str) and value in concept_map:
            return {
                **item,
                "value": concept_map[value],
                "transformation": f"concept mapping: {value} → {concept_map[value]}"
            }
        
        return None

    def select_best_transfer(self, transfers: List[Dict]) -> Optional[Dict]:
        """Select the best knowledge transfer based on mapping score"""
        if not transfers:
            return None
            
        return max(
            transfers, 
            key=lambda t: t["mapping_score"]
        )

    def _format_results(self, best_transfer: Optional[Dict], analogies: List[Dict], context: Dict) -> Dict[str, Any]:
        """Format final results with metadata"""
        return {
            "best_transfer": best_transfer,
            "alternative_analogies": analogies,
            "context_used": context,
            "metrics": {
                "analogies_considered": len(analogies),
                "min_similarity": self.min_similarity,
                "mapping_score": best_transfer["mapping_score"] if best_transfer else 0,
                "success": best_transfer is not None
            },
            "reasoning_type": "analogical"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Analogical ===\n")
    printer.status("TEST", "Starting Reasoning Analogical tests", "info")

    analogy = ReasoningAnalogical()

    # Create test domains
    target = {
        "name": "Smart City Transportation",
        "features": ["traffic management", "public transit", "sustainability"],
        "challenges": ["congestion", "emissions"],
        "goals": ["efficiency", "accessibility"]
    }
    
    source_domain = [
        {
            "name": "Ecosystem Management",
            "features": ["biodiversity", "resource cycling", "sustainability"],
            "solutions": ["keystone species", "balanced populations"],
            "principles": ["balance", "diversity"]
        },
        {
            "name": "Computer Network",
            "features": ["traffic routing", "bandwidth allocation", "security"],
            "solutions": ["QoS protocols", "firewalls"],
            "principles": ["efficiency", "reliability"]
        }
    ]
    
    context = {
        "constraints": {
            "solutions": {"scale_factor": 0.8}
        },
        "concept_mappings": {
            "keystone species": "key infrastructure",
            "balanced populations": "balanced modal share"
        }
    }

    result = analogy.perform_reasoning(
        target=target,
        source_domain=source_domain,
        context=context
    )
    
    printer.pretty("Analogical Reasoning Result", result)
    print("\n=== Successfully Ran Reasoning Analogical ===\n")