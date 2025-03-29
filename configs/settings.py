# config/settings.py

# Environment configuration
ENVIRONMENT_CONFIG = {
    'name': 'Unreal Tournament',
    'type': 'game',
    'update_interval': 0.1  # seconds
}

# Agent configurations
AGENT_CONFIG = {
    'agent1': {
        'type': 'hybrid',
        'name': 'UASTroop',
        'team': 'blue',
        'learning_enabled': True
    },
    'agent2': {
        'type': 'learning',
        'name': 'SupportBot',
        'team': 'blue',
        'learning_enabled': True
    }
}
