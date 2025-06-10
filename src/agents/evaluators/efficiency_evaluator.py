import numpy as np
import yaml, json
import torch
import time
import sys
if sys.platform != "win32":
    import resource
else:
    resource = None

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Any, Dict, Optional
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize

from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Efficiency Evaluator")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

DEFAULT_WEIGHTS = {}

class EfficiencyEvaluator:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('efficiency_evaluator', {})
        memory = EvaluatorsMemory(config)
        self.memory = memory
        self.baseline_measurements = self.config.get('baselines', {}) or {}
        self.complexity_metrics = self.config.get('complexity_metrics', True)

        # Initialize NLP components
        self.tokenizer = self._initialize_tokenizer()
        self.nlp_engine = self._initialize_nlp_engine()

        logger.info(f"Efficiency Evaluator succesfully initialized")

    def _initialize_tokenizer(self):
        """Initialize tokenizer from perception config"""
        from src.agents.perception.modules.tokenizer import Tokenizer, load_config as load_perception_config
        try:
            perception_config = load_perception_config()
            return Tokenizer(perception_config)
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {e}")
            return None

    def _initialize_nlp_engine(self):
        """Initialize NLP engine from language config"""
        from src.agents.language.nlp_engine import NLPEngine
        try:
            return NLPEngine()
        except Exception as e:
            logger.error(f"NLP Engine initialization failed: {e}")
            return None

    def evaluate(self, outputs: List[Any], ground_truths: List[Any]) -> Dict[str, float]:
        """Multi-dimensional efficiency assessment with linguistic analysis"""
        start_time = time.perf_counter()
        if resource:
            start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        else:
            start_mem = 0

        # Base metrics
        metrics = {
            'temporal': self._calculate_temporal(outputs),
            'spatial': self._calculate_spatial(outputs),
            'computational': self._calculate_computational(),
            'token_efficiency': self._calculate_token_efficiency(outputs)
        }

        # Advanced linguistic metrics
        if self.complexity_metrics and self.nlp_engine:
            ling_complexity = self._calculate_linguistic_complexity(outputs)
            metrics.update({
                'syntactic_complexity': ling_complexity['dependency_complexity'],
                'semantic_density': ling_complexity['entity_density'],
                'structural_variety': ling_complexity['pos_diversity']
            })

        # Resource monitoring
        if resource:
            end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        else:
            end_mem = 0
            
        metrics.update({
            'memory_usage_mb': (end_mem - start_mem) / 1024,
            'execution_time': time.perf_counter() - start_time,
            'score': self._calculate_composite_score(metrics)
        })

        if self.config.get('store_results', False):
            self.memory.add(
                entry=metrics,
                tags=["efficiency_eval"],
                priority="medium"
            )
            if self.memory.access_counter % self.memory.config.get("checkpoint_freq", 500) == 0:
                self.memory.create_checkpoint()

        return metrics

    def _calculate_linguistic_complexity(self, outputs: List[Any]) -> Dict[str, float]:
        """Analyze text complexity using NLP Engine"""
        complexity = {
            'avg_sentence_length': 0.0,
            'pos_diversity': 0.0,
            'dependency_complexity': 0.0,
            'entity_density': 0.0
        }
        
        if not self.nlp_engine:
            return complexity

        total_sentences = 0
        pos_counts = defaultdict(int)
        total_dependencies = 0
        total_entities = 0
        total_tokens = 0

        for output in outputs:
            if not isinstance(output, str):
                continue
                
            try:
                # Process text through full NLP pipeline
                tokens = self.nlp_engine.process_text(output)
                sentences = [tokens]  # Simple sentence split
                deps = self.nlp_engine.apply_dependency_rules(tokens)
                entities = self.nlp_engine.resolve_coreferences(sentences)

                # Update metrics
                total_sentences += len(sentences)
                total_tokens += len(tokens)
                total_dependencies += len(deps)
                total_entities += len(entities)
                
                for token in tokens:
                    pos_counts[token.pos] += 1

            except Exception as e:
                logger.warning(f"Complexity analysis failed for output: {e}")

        # Calculate final metrics
        if total_sentences > 0:
            complexity['avg_sentence_length'] = total_tokens / total_sentences
            
        if pos_counts:
            complexity['pos_diversity'] = len(pos_counts) / total_tokens
            
        if total_tokens > 0:
            complexity['dependency_complexity'] = total_dependencies / total_tokens
            complexity['entity_density'] = total_entities / total_tokens

        return complexity

    def _calculate_token_efficiency(self, outputs: List[Any]) -> float:
        """Enhanced token efficiency with subword awareness"""
        total_tokens = 0
        
        for o in outputs:
            text = str(o)
            if self.tokenizer:
                try:
                    encoded = self.tokenizer.encode(text)
                    total_tokens += len(encoded['input_ids'])
                except:
                    total_tokens += len(text.split())
            else:
                total_tokens += len(text.split())
        
        avg_tokens = total_tokens / len(outputs) if outputs else 0
        baseline = self.config.get('nlg', {}).get('avg_token_baseline', 50)
        return baseline / (avg_tokens + sys.float_info.epsilon)

    def _calculate_temporal(self, outputs: List[Any]) -> float:
        """Response latency efficiency using generation time metadata.
        
        Steps:
        1. Extract generation time from output metadata
        2. Use baseline latency per output type
        3. Calculate efficiency ratio with smoothing
        """
        # Fallback if no temporal data exists
        if not any(isinstance(o, dict) and 'generation_time' in o for o in outputs):
            logger.warning("Temporal efficiency: No generation_time metadata found")
            return 0.0  # Missing data penalty
    
        total_time = 0.0
        valid_outputs = 0
        
        for output in outputs:
            if isinstance(output, dict) and 'generation_time' in output:
                total_time += output['generation_time']
                valid_outputs += 1
        
        # Get baseline from config (per-output or batch)
        time_baseline = self.baseline_measurements.get('time_per_output', 0.5)
        if 'time_baseline' in self.baseline_measurements:
            baseline = self.baseline_measurements['time_baseline']  # Batch baseline
        else:
            baseline = time_baseline * valid_outputs  # Per-output baseline
            
        # Calculate efficiency ratio with smoothing
        return baseline / (total_time + sys.float_info.epsilon)

    def _calculate_spatial(self, outputs: List[Any]) -> float:
        """Memory footprint efficiency with serialization.
        
        Steps:
        1. Measure serialized size of outputs
        2. Compare against content-aware baselines
        3. Handle different output types
        """
        total_size = 0
        content_types = defaultdict(int)
        
        for output in outputs:
            # Prefer metadata if available
            if isinstance(output, dict) and 'serialized_size' in output:
                total_size += output['serialized_size']
                content_types[output.get('content_type', 'unknown')] += 1
                continue
                
            try:
                # Serialize and measure payload size
                if isinstance(output, (str, bytes)):
                    serialized = output
                else:
                    serialized = json.dumps(output).encode('utf-8')
                total_size += len(serialized)
                content_types['structured'] += 1
            except (TypeError, OverflowError):
                # Fallback for non-serializable objects
                total_size += sys.getsizeof(output)
                content_types['binary'] += 1
        
        # Get baseline based on content profile
        baseline = self.baseline_measurements.get('memory_baseline', 1024)  # Default 1KB
        if 'memory_per_type' in self.baseline_measurements:
            # Calculate type-aware baseline
            type_baselines = self.baseline_measurements['memory_per_type']
            baseline = sum(
                type_baselines.get(ctype, type_baselines.get('default', 512)) * count 
                for ctype, count in content_types.items()
            )
        
        # Calculate efficiency ratio
        return baseline / (total_size + sys.float_info.epsilon)

    def _calculate_computational(self):
        """FLOPs relative to baseline"""
        baseline = float(self.baseline_measurements.get('flops', 1e6))
        current = float(self.config.get('current_flops', baseline))
        return baseline / (current + sys.float_info.epsilon)

    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Context-aware scoring with linguistic weights"""
        weights = self.config.get('efficiency_weights', {
            'temporal': 0.3,
            'spatial': 0.2,
            'computational': 0.2,
            'token_efficiency': 0.1,
            'syntactic_complexity': 0.1,
            'semantic_density': 0.1
        })

        return sum(
            weights.get(metric, 0) * min(value, 1.0)
            for metric, value in metrics.items()
            if not metric.startswith('memory') and metric != 'execution_time'
        )

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate comprehensive natural language efficiency report"""
        from src.agents.evaluators.utils.report import get_visualizer
        report = []
    
        visualizer = get_visualizer()
        visualizer.update_metrics({
            'successes': metrics.get('successes', 0),
            'computational': metrics.get('computational', 0)
        })
        

        # Header Section
        report.append(f"# Efficiency Evaluation Report\n")
        report.append(f"**Generated**: {datetime.now().isoformat()}\n")
        chart = visualizer.render_temporal_chart(QSize(600, 400), 'computational')
        report.append(f"![Efficiency Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")

        # Executive Summary using Transformer Generator
        summary = self._generate_executive_summary(metrics)
        report.append(f"\n## Executive Summary\n{summary}")

        report.append(self._generate_metric_analysis(metrics))                     # Core Metrics Analysis
        if self.complexity_metrics and self.nlp_engine:
            report.append(self._generate_linguistic_analysis(metrics))             # Linguistic Insights
        report.append(self._generate_comparative_analysis(metrics))                # Comparative Analysis
        report.append(self._generate_recommendations(metrics))                     # Recommendations
        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")   # Footer with system info
        
        return "\n".join(report)

    def _generate_executive_summary(self, metrics: Dict) -> str:
        """Generate summary using transformer-based generator with proper configuration"""
        from src.agents.perception.encoders.text_encoder import TextEncoder
        from src.agents.perception.modules.transformer import Generator
        from src.agents.perception.modules.tokenizer import Tokenizer
    
        # Load proper configuration
        perception_config = load_config("src/agents/perception/configs/perception_config.yaml")
        
        # Initialize tokenizer with full config
        tokenizer = Tokenizer(perception_config)
        
        # Transformer configuration with all required parameters
        transformer_config = {
            'transformer': perception_config['transformer'],
            'attention': perception_config['attention'],
            'feedforward': perception_config['feedforward'],
            'text_encoder': perception_config['text_encoder'],
            'tokenizer': perception_config['tokenizer']
        }
    
        # Initialize text encoder with proper device mapping
        text_encoder = TextEncoder(
            config=transformer_config,
            device='cpu',
            tokenizer=tokenizer
        )
        
        # Ensure hidden_dim matches text_encoder's embed_dim
        hidden_dim = text_encoder.embed_dim
    
        # Initialize generator with validated parameters
        generator = Generator(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=text_encoder.embed_dim
        )
    
        # Create properly formatted prompt with metric data
        prompt = (
            "Generate an executive summary for an efficiency report with these metrics:\n"
            f"- Overall score: {metrics.get('score', 0):.2f}/1.0\n"
            f"- Temporal efficiency: {metrics.get('temporal', 0):.2f}\n"
            f"- Semantic density: {metrics.get('semantic_density', 0):.2f}\n"
            "Focus on key strengths and areas needing improvement:" # . Use concise, professional language:"
        )
    
        # Generate with proper attention mask dtype
        return generator.generate(
            prompt=prompt,
            max_length=200,
            temperature=0.7,
            top_k=25,
            top_p=0.9,
            # attention_mask=torch.ones(len(prompt.split())),  # Create boolean mask
            # causal=True
        )

    def _describe_complexity(self, score: float) -> str:
        """Convert complexity score to descriptive text using tokenizer"""
        descriptors = {
            0: "very low",
            1: "low",
            2: "moderate",
            3: "high",
            4: "very high"
        }
        
        # Use tokenizer to process descriptor if available
        if self.tokenizer:
            encoded = self.tokenizer.encode(str(score))
            quantized = min(len(descriptors)-1, int(encoded['input_ids'][0] % 5))
            return descriptors.get(quantized, "moderate")
        
        # Fallback linear mapping
        return descriptors.get(min(4, int(score * 4)), "moderate")

    def _generate_metric_analysis(self, metrics: Dict) -> str:
        """Detailed metric breakdown with visual indicators"""
        analysis = ["\n## Core Metrics Analysis\n"]
        
        metric_map = {
            'temporal': ('⏱ Temporal Efficiency', 'Response latency efficiency score'),
            'spatial': ('💾 Spatial Efficiency', 'Memory utilization efficiency'),
            'computational': ('⚡ Computational Efficiency', 'FLOPs relative to baseline'),
            'token_efficiency': ('🔠 Token Efficiency', 'Output compactness score')
        }
        
        for metric, (title, desc) in metric_map.items():
            value = metrics.get(metric, 0)
            bar = '█' * int(value * 20) + '-' * (20 - int(value * 20))
            analysis.append(
                f"### {title}\n"
                f"{desc}\n"
                f"**Score**: {value:.2f}/1.0  \n"
                f"`{bar}`\n"
            )
            
        return "\n".join(analysis)

    def _load_report_template(self, template_path: Path) -> Dict:
        """Load structured report template"""
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load report template: {e}")
        return {
            'sections': [
                'executive_summary',
                'metric_analysis',
                'linguistic_insights',
                'comparisons',
                'recommendations'
            ]
        }
        
    def _generate_linguistic_analysis(self, metrics: Dict) -> str:
        """Enhanced linguistic analysis using NLP Engine features"""
        
        analysis = ["\n## Advanced Linguistic Analysis\n"]
        
        if not self.nlp_engine:
            analysis.append("*NLP features unavailable - engine not initialized*")
            return "\n".join(analysis)
    
        # Basic metrics from config
        analysis.append(
            f"- **Syntactic Complexity**: {metrics.get('syntactic_complexity', 0):.2f}\n"
            f"  (Dependencies per token, higher = more complex structures)\n"
            f"- **Semantic Density**: {metrics.get('semantic_density', 0):.2f}\n"
            f"  (Entities per token, higher = more information density)\n"
            f"- **Structural Variety**: {metrics.get('structural_variety', 0):.2f}\n"
            f"  (POS diversity ratio, higher = more grammatical variation)\n"
        )
    
        # Advanced NLP diagnostics
        advanced_metrics = {
            'sarcasm_score': 0.0,
            'coref_chains': 0,
            'avg_mentions': 0.0,
            'dependency_types': defaultdict(int)
        }
        total_outputs = 0
        
        # Sample processing of outputs (assuming access via self.memory or other source)
        try:
            outputs = self.memory.get_last_entries(limit=50)  # Get recent outputs for analysis
            for entry in outputs:
                if 'output' not in entry:
                    continue
                    
                text = str(entry['output'])
                tokens = self.nlp_engine.process_text(text)
                deps = self.nlp_engine.apply_dependency_rules(tokens)
                entities = self.nlp_engine.resolve_coreferences([tokens])
                
                # Sarcasm detection
                advanced_metrics['sarcasm_score'] += self.nlp_engine.detect_sarcasm(tokens)
                
                # Coreference analysis
                coref_clusters = defaultdict(list)
                for ent in entities:
                    coref_clusters[ent.coref_id].append(ent)
                advanced_metrics['coref_chains'] += len(coref_clusters)
                if coref_clusters:
                    advanced_metrics['avg_mentions'] += sum(len(c) for c in coref_clusters.values())/len(coref_clusters)
                
                # Dependency diversity
                for dep in deps:
                    advanced_metrics['dependency_types'][dep.relation] += 1
                    
                total_outputs += 1
                
        except Exception as e:
            logger.error(f"Advanced linguistic analysis failed: {e}")
    
        if total_outputs > 0:
            # Normalize metrics
            advanced_metrics['sarcasm_score'] /= total_outputs
            advanced_metrics['coref_chains'] = advanced_metrics['coref_chains']/total_outputs
            advanced_metrics['avg_mentions'] = advanced_metrics['avg_mentions']/total_outputs
            
            # Add NLP-powered insights
            analysis.append("### Discourse Features\n")
            analysis.append(
                f"- **Sarcasm Likelihood**: {advanced_metrics['sarcasm_score']:.2f}/1.0\n"
                f"  (Higher values indicate more potential ironic/sarcastic content)\n"
                f"- **Coreference Chains**: {advanced_metrics['coref_chains']:.1f} per output\n"
                f"  (Entities needing tracking across mentions)\n"
                f"- **Mentions per Chain**: {advanced_metrics['avg_mentions']:.1f}\n"
                f"  (Higher values indicate more complex reference tracking)\n"
            )
            
            # Dependency analysis
            analysis.append("### Syntactic Patterns\n")
            total_deps = sum(advanced_metrics['dependency_types'].values())
            if total_deps > 0:
                common_deps = sorted(
                    advanced_metrics['dependency_types'].items(),
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]  # Top 3 most common
                
                analysis.append("Most frequent dependency relations:\n")
                for dep_type, count in common_deps:
                    analysis.append(f"  - {dep_type.title()} ({count/total_deps:.1%})\n")
            else:
                analysis.append("*No dependency relations found*\n")
    
        else:
            analysis.append("*No textual outputs available for advanced analysis*\n")
    
        return "\n".join(analysis)
    
    def _generate_comparative_analysis(self, metrics: Dict) -> str:
        """Benchmark comparisons using configured baselines"""
        analysis = ["\n## Comparative Analysis\n"]
        
        comparisons = {
            'flops': ('FLOPs', 'computational', 'current_flops'),
            'memory': ('Memory Usage', 'spatial', 'memory_threshold')
        }
        
        for key, (name, metric, baseline_key) in comparisons.items():
            current = metrics.get(metric, 0)
            baseline = self.baseline_measurements.get(baseline_key, 1)
            ratio = current / baseline if baseline else 0
            
            analysis.append(
                f"- **{name}**: {current:.2f} vs baseline {baseline}  \n"
                f"  ({'✅ Meets' if ratio >= 1 else '❌ Below'} efficiency target)\n"
            )
            
        return "\n".join(analysis)
    
    def _generate_recommendations(self, metrics: Dict) -> str:
        """Actionable recommendations loaded from JSON template"""
        template_path = Path("src/agents/evaluators/templates/recommendation_template.json")
        try:
            with open(template_path, 'r') as f:
                template = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load recommendations template: {e}")
            return "\n## Recommendations Unavailable\n"
        
        recommendations = []
        for section in template.get("recommendation_sections", []):
            section_recs = []
            for rec in section.get("recommendations", []):
                condition = rec.get("condition", {})
                metric_key = condition.get("metric")
                op = condition.get("operator")
                threshold = condition.get("threshold")
                
                if metric_key not in metrics:
                    continue  # Skip invalid/unavailable metrics
                
                current_value = metrics[metric_key]
                condition_met = False
                
                # Evaluate condition
                if op == "<":
                    condition_met = (current_value < threshold)
                elif op == ">":
                    condition_met = (current_value > threshold)
                elif op == "<=":
                    condition_met = (current_value <= threshold)
                elif op == ">=":
                    condition_met = (current_value >= threshold)
                
                if condition_met:
                    section_recs.append(f"- {rec['message']}")
            
            if section_recs:
                recommendations.append(
                    f"\n### {section['section_title']}\n" + "\n".join(section_recs)
                )
        
        if not recommendations:
            return "\n## No Critical Issues Detected\n"
        
        return "\n## Optimization Recommendations\n" + "\n".join(recommendations)

    def disable_temporarily(self):
        """Temporarily disable efficiency testing during degraded mode"""
        self.test_cases = []
        logger.warning("Efficiency Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Efficiency Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    config = load_config()

    eval = EfficiencyEvaluator(config)

    logger.info(f"{eval}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = [4]
    ground_truths = [5]

    results = eval.evaluate(outputs, ground_truths)
    logger.info(f"Evaluation results: {results}")
    print(f"\n* * * * * Phase 3 * * * * *\n")
    outputs = ["The quick brown fox jumps over the lazy dog.", 
               "Sample output for efficiency analysis."]
    ground_truths = ["Expected reference text for comparison."]

    report = eval.generate_report(results)
    logger.info(f"Evaluation report: {report}")
    print("\n=== Successfully Ran Efficiency Evaluator ===\n")
