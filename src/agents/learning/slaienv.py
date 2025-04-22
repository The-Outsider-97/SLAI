class SLAIEnv:
    """Base environment interface for SLAI operations"""
    def __init__(self, SLAILM, agent_factory, shared_memory, state_dim=4, action_dim=2, env=None):
        self.observation_space = self.ObservationSpace(state_dim)
        self.action_space = self.ActionSpace(action_dim)
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.SLAILM = SLAILM
        self.env = env

    class ObservationSpace:
        def __init__(self, dim):
            self.shape = [dim]
            self.low = [-1.0] * dim
            self.high = [1.0] * dim
            self.dtype = float

        def sample(self):
            import random
            return [random.uniform(l, h) for l, h in zip(self.low, self.high)]

    class ActionSpace:
        def __init__(self, n):
            self.n = n
            self.actions = list(range(n))

        def sample(self):
            import random
            return random.choice(self.actions)

    def reset(self):
        if self.env and hasattr(self.env, 'reset'):
            return self.env.reset()
        # Provide dummy state
        return [0.0] * self.observation_space.shape[0]

    def step(self, action):
        if self.env and hasattr(self.env, 'step'):
            return self.env.step(action)
        # Dummy step: return next state, reward, done, info
        next_state = [0.0] * self.observation_space.shape[0]
        reward = 0.0
        done = True
        info = {}
        return next_state, reward, done, info
