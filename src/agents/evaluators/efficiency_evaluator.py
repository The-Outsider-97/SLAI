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

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_errors import (
    EvaluationError, ConfigLoadError, ValidationFailureError,
    MetricCalculationError, ReportGenerationError, TemplateError,
    VisualizationError, MemoryAccessError
)
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Efficiency Evaluator")
printer = PrettyPrinter

DEFAULT_WEIGHTS = {}

class EfficiencyEvaluator:
    def __init__(self):
        self.config = load_global_config()
        self.trans_config = get_config_section('efficiency_evaluator')
        self.energy_model = self.trans_config.get('energy_model')
        self.report_template = self.trans_config.get('report_template')
        self.complexity_metrics = self.trans_config.get('complexity_metrics')
        self.recommendation_template_path = self.trans_config.get('recommendation_template_path')
        self.current_flops = float(self.trans_config.get('current_flops'))
        self.efficiency_weights = self.trans_config.get('efficiency_weights', {})
        self.linguistic_weights = self.trans_config.get('linguistic_weights', {})
        self.baselines = {
            k: float(v) if isinstance(v, (int, float, str)) else v 
            for k, v in self.trans_config.get('baselines', {}).items()
        }

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        self.tokenizer = None
        self.nlp_engine = None

        logger.info(f"Efficiency Evaluator successfully initialized")

    def _initialize_nlp_engine(self):
        """Initialize NLP engine from language config"""
        try:
            from src.agents.language.nlp_engine import NLPEngine
            self.nlp_engine = NLPEngine()
            return self.nlp_engine
        except ImportError as e:
            raise ConfigLoadError(
                config_path="language_config",
                section="nlp_engine",
                error_details=f"NLP Engine import failed: {str(e)}"
            )
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

        metrics = {}
        try:
            # Base metrics
            metric_functions = {
                'temporal': self.calculations._calculate_temporal,
                'spatial': self.calculations._calculate_spatial,
                'computational': lambda _: self.calculations._calculate_computational(),
                'token_efficiency': self.calculations._calculate_token_efficiency
            }
            
            for metric_name, metric_fn in metric_functions.items():
                try:
                    metrics[metric_name] = metric_fn(outputs)
                except Exception as e:
                    raise MetricCalculationError(
                        metric_name=metric_name,
                        inputs=outputs[:5],
                        reason=str(e)
                    )

            # Advanced linguistic metrics
            if self.complexity_metrics:
                if not self.nlp_engine:
                    self._initialize_nlp_engine()
                    
                if self.nlp_engine:
                    try:
                        ling_complexity = self.calculations._calculate_linguistic_complexity(outputs)
                        metrics.update({
                            'syntactic_complexity': ling_complexity['dependency_complexity'],
                            'semantic_density': ling_complexity['entity_density'],
                            'structural_variety': ling_complexity['pos_diversity']
                        })
                    except Exception as e:
                        raise MetricCalculationError(
                            metric_name="linguistic_complexity",
                            inputs=outputs[:5],
                            reason=str(e)
                        )
                else:
                    logger.warning("NLP engine unavailable, skipping linguistic metrics")

            # Resource monitoring
            if resource:
                end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            else:
                end_mem = 0
                
            metrics.update({
                'memory_usage_mb': (end_mem - start_mem) / 1024,
                'execution_time': time.perf_counter() - start_time
            })
            
            # Calculate composite score
            try:
                # metrics['score'] = self.calculations._calculate_composite_score(metrics)
                composite_score = 0.0
                for metric, value in metrics.items():
                    if metric in self.efficiency_weights:
                        composite_score += self.efficiency_weights[metric] * value
                metrics['score'] = composite_score
            except Exception as e:
                raise MetricCalculationError(
                    metric_name="composite_score",
                    inputs=metrics,
                    reason=str(e)
                )

            if self.config.get('store_results', False):
                try:
                    self.memory.add(
                        entry=metrics,
                        tags=["efficiency_eval"],
                        priority="medium"
                    )
                    if self.memory.access_counter % self.memory.config.get("checkpoint_freq", 500) == 0:
                        self.memory.create_checkpoint()
                except Exception as e:
                    raise MemoryAccessError(
                        operation="add",
                        key="efficiency_metrics",
                        error_details=str(e)
                    )

            return metrics
        except EvaluationError as e:
            logger.error(f"Efficiency evaluation failed: {e.to_audit_dict()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during efficiency evaluation: {str(e)}")
            raise MetricCalculationError(
                metric_name="efficiency_metrics",
                inputs={"outputs": outputs[:5], "ground_truths": ground_truths[:5]},
                reason=str(e)
            )

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate comprehensive natural language efficiency report"""
        try:
            report = []
            visualizer = get_visualizer()
            
            # Validate required metrics
            required_metrics = ['temporal', 'spatial', 'computational', 'token_efficiency', 'score']
            for metric in required_metrics:
                if metric not in metrics:
                    raise ValidationFailureError(
                        rule_name="required_metric",
                        data=metrics.keys(),
                        expected=f"Metric '{metric}' must be present"
                    )

            try:
                visualizer.update_metrics({
                    'successes': metrics.get('successes', 0),
                    'computational': metrics.get('computational', 0)
                })
            except Exception as e:
                raise VisualizationError(
                    chart_type="temporal",
                    data=metrics,
                    error_details=f"Failed to update visualizer metrics: {str(e)}"
                )

            # Header Section
            report.append(f"# Efficiency Evaluation Report\n")
            report.append(f"**Generated**: {datetime.now().isoformat()}\n")
            try:
                chart = visualizer.render_temporal_chart(QSize(600, 400), 'computational')
                report.append(f"![Efficiency Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")
            except Exception as e:
                raise VisualizationError(
                    chart_type="temporal",
                    data=metrics,
                    error_details=f"Failed to render chart: {str(e)}"
                )

            # Executive Summary using Transformer Generator
            try:
                summary = self._generate_executive_summary(metrics)
                report.append(f"\n## Executive Summary\n{summary}")
            except Exception as e:
                report.append("\n## Executive Summary\n*Could not generate summary*")
                logger.error(f"Executive summary generation failed: {str(e)}")

            # Core sections with error handling
            try:
                report.append(self._generate_metric_analysis(metrics))
            except Exception as e:
                report.append("\n## Core Metrics Analysis\n*Metric analysis failed*")
                logger.error(f"Metric analysis failed: {str(e)}")
                
            if self.complexity_metrics and self.nlp_engine:
                try:
                    report.append(self._generate_linguistic_analysis(metrics))
                except Exception as e:
                    report.append("\n## Linguistic Analysis\n*Linguistic analysis failed*")
                    logger.error(f"Linguistic analysis failed: {str(e)}")
                    
            try:
                report.append(self._generate_comparative_analysis(metrics))
            except Exception as e:
                report.append("\n## Comparative Analysis\n*Comparative analysis failed*")
                logger.error(f"Comparative analysis failed: {str(e)}")
                
            try:
                report.append(self._generate_recommendations(metrics))
            except Exception as e:
                report.append("\n## Recommendations\n*Recommendations generation failed*")
                logger.error(f"Recommendations generation failed: {str(e)}")
                
            report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
            
            return "\n".join(report)
        except EvaluationError as e:
            logger.error(f"Report generation error: {e.to_audit_dict()}")
            raise ReportGenerationError(
                report_type="Efficiency",
                template=self.report_template or "efficiency_report",
                error_details=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during report generation: {str(e)}")
            raise ReportGenerationError(
                report_type="Efficiency",
                template=self.report_template or "efficiency_report",
                error_details=f"Unexpected error: {str(e)}"
            )

    def _generate_executive_summary(self, metrics: Dict) -> str:
        """Generate summary using transformer-based generator with proper configuration"""
        try:
            from src.agents.perception.encoders.text_encoder import TextEncoder
            from src.agents.perception.decoders.text_decoder import TextDecoder
            from src.agents.perception.modules.tokenizer import Tokenizer

            tokenizer = Tokenizer()
            text_encoder = TextEncoder().to('cuda' if torch.cuda.is_available() else 'cpu')
            text_decoder = TextDecoder(encoder=text_encoder).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create properly formatted prompt with metric data
            prompt = (
                "Generate an executive summary for an efficiency report with these metrics:\n"
                f"- Overall score: {metrics.get('score', 0):.2f}/1.0\n"
                f"- Temporal efficiency: {metrics.get('temporal', 0):.2f}\n"
                f"- Semantic density: {metrics.get('semantic_density', 0):.2f}\n"
                "Focus on key strengths and areas needing improvement:"
            )
            
            # Tokenize and encode the prompt
            prompt_encoding = tokenizer(prompt)
            input_ids = prompt_encoding["input_ids"]
            attention_mask = prompt_encoding["attention_mask"]
            
            # Move tensors to the same device as the encoder
            device = next(text_encoder.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Add batch dimension
            input_ids = input_ids.unsqueeze(0)  # Shape: [1, seq_len]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, seq_len]
            
            # Encode prompt to create memory
            encoded_prompt = text_encoder(input_ids, attention_mask=attention_mask)
            
            # Generate summary using decoder inference
            generated_ids = text_decoder.inference(
                memory=encoded_prompt,
                strategy="sampling",
                temperature=0.7,
                top_k=25,
                top_p=0.9
            )
            
            # Decode generated token IDs to text
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        except ImportError as e:
            raise ConfigLoadError(
                config_path="text_generation",
                section="text_encoder",
                error_details=f"Required module not found: {str(e)}"
            )
        except Exception as e:
            # Fallback to simple template
            return (
                f"Efficiency Score: {metrics.get('score', 0):.2f}/1.0\n"
                f"Temporal Efficiency: {metrics.get('temporal', 0):.2f}\n"
                f"Computational Efficiency: {metrics.get('computational', 0):.2f}\n"
                f"Memory Usage: {metrics.get('memory_usage_mb', 0):.2f} MB\n"
                "Recommendation: Focus on optimizing computational efficiency"
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
            try:
                encoded = self.tokenizer.encode(str(score))
                quantized = min(len(descriptors)-1, int(encoded['input_ids'][0] % 5))
                return descriptors.get(quantized, "moderate")
            except Exception:
                pass
        
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
            except json.JSONDecodeError as e:
                raise TemplateError(
                    template_path=str(template_path),
                    error_details=f"Invalid JSON format: {str(e)}"
                )
            except Exception as e:
                raise TemplateError(
                    template_path=str(template_path),
                    error_details=f"Template loading failed: {str(e)}"
                )
        else:
            raise TemplateError(
                template_path=str(template_path),
                error_details="Template file not found"
            )
        
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
            raise MetricCalculationError(
                metric_name="linguistic_analysis",
                inputs={"outputs": outputs[:3] if outputs else []},
                reason=f"Advanced NLP processing failed: {str(e)}"
            )
    
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
            'flops': ('FLOPs', 'computational', 'flops'),
            'memory': ('Memory Usage', 'memory_usage_mb', 'memory_threshold')
        }
        
        for key, (name, metric, baseline_key) in comparisons.items():
            current = metrics.get(metric, 0)
            
            # Safely get and convert baseline value
            baseline = self.baselines.get(baseline_key, 1.0)
            if not isinstance(baseline, (int, float)):
                try:
                    baseline = float(baseline)
                except (ValueError, TypeError):
                    baseline = 1.0
                    logger.warning(f"Invalid baseline for {baseline_key}: {self.baselines.get(baseline_key)}")
            
            # Handle FLOPs comparison
            if metric == 'computational':
                try:
                    ratio = self.current_flops / baseline if baseline else 0
                    status = "Meets" if ratio <= 1 else "Exceeds"
                    analysis.append(
                        f"- **{name}**: {self.current_flops:.2e} vs baseline {baseline:.2e}\n"
                        f"  ({status} efficiency target)\n"
                    )
                except Exception as e:
                    raise MetricCalculationError(
                        metric_name="flops_comparison",
                        inputs={"current_flops": self.current_flops, "baseline": baseline},
                        reason=str(e)
                    )
            else:
                try:
                    ratio = current / baseline if baseline else 0
                    status = "Meets" if ratio <= 1 else "Exceeds"
                    analysis.append(
                        f"- **{name}**: {current:.2f} vs baseline {baseline:.2f}\n"
                        f"  ({status} efficiency target)\n"
                    )
                except Exception as e:
                    raise MetricCalculationError(
                        metric_name="memory_comparison",
                        inputs={"current": current, "baseline": baseline},
                        reason=str(e)
                    )
                
        return "\n".join(analysis)
    
    def _generate_recommendations(self, metrics: Dict) -> str:
        """Actionable recommendations loaded from JSON template"""
        template_path = Path(self.recommendation_template_path)
        try:
            template = self._load_report_template(template_path)
        except TemplateError as e:
            logger.error(f"Recommendation template error: {str(e)}")
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
    eval = EfficiencyEvaluator()

    logger.info(f"{eval}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = ["The quick brown fox jumps over the lazy dog.", 
               "Sample output for efficiency analysis."]
    ground_truths = ["Expected reference text for comparison."]

    results = eval.evaluate(outputs, ground_truths)
    printer.pretty(f"Evaluation results:", results, "success" if results else "error")
    print(f"\n* * * * * Phase 3 * * * * *\n")


    report = eval.generate_report(results)
    printer.pretty(f"Evaluation report:", report, "success" if report else "error")
    print("\n=== Successfully Ran Efficiency Evaluator ===\n")
