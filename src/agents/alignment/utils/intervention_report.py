

import hashlib
import json
import numpy as np
import pandas as pd

from typing import Dict, Any, List
from datetime import datetime

class InterventionReport:
    def __init__(self, agent: Any):
        """
        Initialize with a reference to the calling agent
        to access alignment components and state
        """
        self.agent = agent
        self.timestamp = datetime.now().isoformat()

    def generate(self) -> Dict[str, Any]:
        """Generate comprehensive intervention report with forensic evidence"""
        return {
            'metadata': self._generate_metadata(),
            'risk_analysis': self._current_risk_assessment(),
            'counterfactuals': self._generate_counterfactual_examples(),
            'violation_timeline': self._construct_violation_timeline(),
            'recommended_actions': self._suggest_potential_fixes(),
            'system_state_fingerprint': self._compute_state_fingerprint(),
            'forensic_snapshot': self._capture_forensic_state()
        }

    def _generate_metadata(self) -> Dict[str, str]:
        return {
            'agent_id': self.agent.agent_id if hasattr(self.agent, 'agent_id') else 'unknown',
            'report_timestamp': self.timestamp,
            'intervention_level': 'CRITICAL',
            'protocol_version': '2.3'
        }

    def _current_risk_assessment(self) -> Dict[str, Any]:
        """Compute current risk profile using agent's existing methods"""
        # Reuse alignment checks if available
        if hasattr(self.agent, 'last_alignment_report'):
            report = self.agent.last_alignment_report
        else:
            # Fallback to latest risk history
            return self.agent.risk_history[-1] if self.agent.risk_history else {}
        
        return self.agent._assemble_risk_profile(report)

    def _generate_counterfactual_examples(self) -> Dict[str, Any]:
        """Generate explainable counterfactuals using agent's auditor"""
        if not hasattr(self.agent, 'auditor'):
            return {}
            
        return self.agent.auditor.generate_explainable_counterfactuals(
            num_examples=3,
            max_iterations=100
        )

    def _construct_violation_timeline(self) -> List[Dict]:
        """Create timeline of recent violations from shared memory"""
        timeline = []
        history = self.agent.shared_memory.get('alignment_history', [])
        
        for record in history[-5:]:  # Last 5 records
            timeline.append({
                'timestamp': record['timestamp'],
                'event_type': 'risk_anomaly',
                'risk_level': record.get('risk_assessment', {}).get('total_risk', 0),
                'violation_types': record.get('ethical_violations_details', [])
            })
        return timeline

    def _suggest_potential_fixes(self) -> Dict[str, Any]:
        """Generate repair suggestions based on violation patterns"""
        corrections = {}
        risk_profile = self._current_risk_assessment()
        
        # Component-specific fixes
        if 'statistical_parity' in risk_profile.get('component_risks', {}):
            corrections['threshold_adjustment'] = (
                "Apply demographic parity threshold adjustment with λ=0.75"
            )
        if 'ethical_violations' in risk_profile.get('component_risks', {}):
            corrections['constraint_update'] = (
                "Add constitutional principle #7 to primary constraint set"
            )
            
        # System-level recommendations
        corrections['system'] = [
            "Increase fairness regularization weight by 30%",
            "Enable subgroup monitoring for sensitive attributes"
        ]
        
        return corrections

    def _compute_state_fingerprint(self) -> str:
        """Create unique hash of critical system state"""
        state = {
            'risk_table': self.agent.risk_table,
            'constraint_weights': self.agent.ethics.constraint_weights,
            'config_hash': hashlib.md5(
                json.dumps(self.agent.config, sort_keys=True).encode()
            ).hexdigest(),
            'timestamp': self.timestamp
        }
        return hashlib.sha256(json.dumps(state).encode()).hexdigest()

    def _capture_forensic_state(self) -> Dict[str, Any]:
        """Preserve critical state for audit trails"""
        return {
            'decision_context': self._get_recent_actions(),
            'memory_snapshot': self.agent.shared_memory.get_latest_snapshot(),
            'ethical_state': self.agent.ethics.get_current_state(),
            'performance_metrics': self.agent.shared_memory.get('performance_metrics', {})
        }

    def _get_recent_actions(self) -> List[Dict]:
        """Capture last 3 decisions for context"""
        return self.agent.shared_memory.get('action_history', [])[-3:]
