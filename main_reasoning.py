"""
main_reasoning.py - Entry point for the Cognitive Hybrid Reasoning Intelligent Agent System (CHRIS)
"""

import sys
import time
from utils.agent_factory import AgentFactory
from config.settings import AGENT_CONFIG, ENVIRONMENT_CONFIG

class ReasoningSystem:
    def __init__(self):
        """Initialize the reasoning system with agent factory and environment"""
        self.agent_factory = AgentFactory()
        self.agents = {}
        self.environment = None  # Would be initialized with environment interface
        
    def initialize_system(self):
        """Initialize all system components"""
        print("Initializing CHRIS Reasoning System...")
        
        # Initialize environment
        self._initialize_environment()
        
        # Create agents based on configuration
        for agent_name, agent_config in AGENT_CONFIG.items():
            agent_type = agent_config['type']
            self.agents[agent_name] = self.agent_factory.create_agent(
                agent_type, agent_config, self.environment
            )
            print(f"Created {agent_type} agent: {agent_name}")
    
    def _initialize_environment(self):
        """Initialize the environment interface"""
        # This would connect to the actual environment (e.g., Unreal Tournament)
        # For now we'll just create a mock environment
        self.environment = {
            'name': ENVIRONMENT_CONFIG['name'],
            'state': {},
            'sensors': {}
        }
        print(f"Initialized environment: {self.environment['name']}")
    
    def run(self):
        """Main execution loop for the reasoning system"""
        print("Starting reasoning system main loop...")
        
        try:
            while True:
                # Update environment state
                self._update_environment_state()
                
                # Process each agent through the reasoning stages
                for agent_name, agent in self.agents.items():
                    self._process_agent(agent)
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down reasoning system...")
            self.shutdown()
    
    def _update_environment_state(self):
        """Update the environment state from sensors"""
        # In a real implementation, this would read actual sensor data
        # For now we'll just simulate some state changes
        self.environment['state']['timestamp'] = time.time()
        
        # Simulate some environment changes
        if 'counter' not in self.environment['state']:
            self.environment['state']['counter'] = 0
        self.environment['state']['counter'] += 1
    
    def _process_agent(self, agent):
        """Process an agent through the cognitive hybrid reasoning stages"""
        # Observation Stage
        agent.observe(self.environment)
        
        # Orientation Stage
        agent.orient()
        
        # Decision Stage
        agent.decide()
        
        # Action Stage
        actions = agent.act()
        
        # Learning Stage (if applicable)
        if hasattr(agent, 'learn'):
            agent.learn(self.environment)
        
        # Execute actions in environment
        if actions:
            self._execute_actions(agent, actions)
    
    def _execute_actions(self, agent, actions):
        """Execute the agent's actions in the environment"""
        # In a real implementation, this would interface with the environment
        print(f"Agent {agent.name} executing actions: {actions}")
        
        # Update environment based on actions
        for action in actions:
            if action['type'] == 'movement':
                self.environment['state'][f'{agent.name}_position'] = action['target']
            elif action['type'] == 'communication':
                # Handle inter-agent communication
                pass
    
    def shutdown(self):
        """Cleanup system resources"""
        print("Cleaning up resources...")
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        print("System shutdown complete.")

def main():
    """Main entry point for the reasoning system"""
    reasoning_system = ReasoningSystem()
    reasoning_system.initialize_system()
    reasoning_system.run()

if __name__ == "__main__":
    main()
