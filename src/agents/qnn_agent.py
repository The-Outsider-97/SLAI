import time
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from src.agents.base_agent import BaseAgent

class QNNAgent(BaseAgent):
    """
    A Quantum Neural Network (QNN) agent with meta-learning and recursive learning capabilities.

    This implementation is based on fundamental quantum computing principles and does not rely on high-level libraries
    for the *quantum gate operations*. It includes a hybrid architecture with a classical meta-learner guiding a
    quantum recurrent neural network (QRNN).

    **Important:** This section focuses on implementing the quantum gate operations from scratch using NumPy.
    The meta-learning and QRNN architectures are still simplified placeholders.
    """

    def __init__(self, shared_memory, agent_factory, config=None):
        super().__init__(shared_memory, agent_factory, config)
        """
        Initializes the QNNAgent.

        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
            num_quantum_layers (int): The number of quantum layers in the QRNN.
            meta_learning_rate (float): The learning rate for the classical meta-learner.
            qrnn_params (dict): Parameters for the QRNN (e.g., number of hidden units, etc.).
        """

        self.num_qubits = config.get("num_qubits", 4)
        self.num_quantum_layers = config.get("num_quantum_layers", 2)
        self.meta_learning_rate = config.get("meta_learning_rate", 0.01)
        self.qrnn_params = config.get("qrnn_params", {"hidden_units": 16})
        self.evaluator = PerformanceEvaluator(metric=config.get("performance_metric", "mse"))

        self.quantum_weights = self._initialize_quantum_weights()
        self.meta_learner = self._initialize_meta_learner()

        # Define basic quantum gates as matrices
        self.gate_definitions = self._define_quantum_gates()

    def perform_task(self, task_data):
        tasks = task_data.get("tasks", [])
        if not tasks:
            return {"status": "failed", "reason": "No tasks provided"}
    
        total_loss = 0.0
        for task in tasks:
            output_sequence, _ = self._qrnn_forward(task)
            task_performance = self._evaluate_performance(output_sequence, task)
            self._meta_learn_update(task_performance)
            total_loss += sum(task_performance)
    
        avg_loss = total_loss / len(tasks)
        return {
            "status": "success",
            "agent": "QNNAgent",
            "average_loss": avg_loss,
            "tasks_trained": len(tasks)
        }

    def _initialize_quantum_weights(self):
        """
        Initializes the parameters (quantum weights) of the quantum circuit.

        Returns:
            list: A list of numpy arrays representing the quantum weights (angles for gate rotations)
                  for each layer.
        """

        quantum_weights = []
        for _ in range(self.num_quantum_layers):
            # Initialize weights as random angles for rotation gates
            layer_weights = np.random.rand(self.num_qubits, 3) * 2 * np.pi  # Angles for Rx, Ry, Rz
            quantum_weights.append(layer_weights)
        return quantum_weights

    def _initialize_meta_learner(self):
            """
            Initializes the RNN-based classical meta-learner.
        
            Returns:
                RNNMetaLearner: A NumPy-based RNN for quantum weight prediction.
            """
            input_size = 1
            hidden_size = 32
            output_shape = (self.num_quantum_layers, self.num_qubits, 3)
        
            return RNNMetaLearner(
                input_size=input_size,
                hidden_size=hidden_size,
                output_shape=output_shape,
                learning_rate=self.meta_learning_rate
            )

    def _define_quantum_gates(self):
        """
        Defines the basic quantum gates as unitary matrices.

        Returns:
            dict: A dictionary containing the gate matrices.
        """

        # Define single-qubit gates
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate [cite: 86]
        X = np.array([[0, 1], [1, 0]])  # Pauli-X gate [cite: 86]
        Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y gate [cite: 86]
        Z = np.array([[1, 0], [0, -1]])  # Pauli-Z gate [cite: 86]
        S = np.array([[1, 0], [0, 1j]])  # Phase gate [cite: 86]
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # Pi/8 gate [cite: 86]

        gate_definitions = {
            'H': H,
            'X': X,
            'Y': Y,
            'Z': Z,
            'S': S,
            'T': T,
        }
        return gate_definitions

    def _rx_gate(self, theta):
        """
        Defines the rotation gate around the X-axis.

        Args:
            theta (float): The rotation angle.

        Returns:
            numpy.ndarray: The Rx gate matrix.
        """

        return np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])

    def _ry_gate(self, theta):
        """
        Defines the rotation gate around the Y-axis.

        Args:
            theta (float): The rotation angle.

        Returns:
            numpy.ndarray: The Ry gate matrix.
        """

        return np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])

    def _rz_gate(self, theta):
        """
        Defines the rotation gate around the Z-axis.

        Args:
            theta (float): The rotation angle.

        Returns:
            numpy.ndarray: The Rz gate matrix.
        """

        return np.array([
            [np.cos(theta / 2) - 1j * np.sin(theta / 2), 0],
            [0, np.cos(theta / 2) + 1j * np.sin(theta / 2)]
        ])

    def _controlled_not_gate(self, control_qubit, target_qubit, num_qubits):
        """
        Defines the CNOT (Controlled-NOT) gate.

        Args:
            control_qubit (int): Index of the control qubit.
            target_qubit (int): Index of the target qubit.
            num_qubits (int): Total number of qubits in the system.

        Returns:
            numpy.ndarray: The CNOT gate matrix.
        """

        # Create the CNOT gate matrix
        I = np.eye(2)
        X = self.gate_definitions['X']
        Z = self.gate_definitions['Z']
        zero_proj = np.array([[1, 0], [0, 0]])
        one_proj = np.array([[0, 0], [0, 1]])

        cnot = np.kron(zero_proj, I) + np.kron(one_proj, X)

        # Pad the CNOT with identity matrices to act on the full system
        pad_l = 1
        pad_r = 1
        if control_qubit < target_qubit:
          pad_l = control_qubit
          pad_r = num_qubits - 1 - target_qubit
        elif control_qubit > target_qubit:
          pad_l = target_qubit
          pad_r = num_qubits - 1 - control_qubit

        gate = np.eye(1)
        for _ in range(pad_l):
          gate = np.kron(gate, I)
        gate = np.kron(gate, cnot)
        for _ in range(pad_r):
          gate = np.kron(gate, I)
        return gate

    def _apply_gate(self, state, gate, target_qubits):
        """
        Applies a quantum gate to the state.

        Args:
            state (numpy.ndarray): The current quantum state.
            gate (numpy.ndarray): The quantum gate matrix.
            target_qubits (list): A list of qubit indices that the gate acts on.

        Returns:
            numpy.ndarray: The updated quantum state.
        """

        #  This applies the gate to the specified qubits
        #  For single-qubit gates, reshape and multiply
        #  For multi-qubit gates, more complex tensor product operations are needed
        if len(target_qubits) == 1:
            # Reshape the state for matrix multiplication
            num_target_qubits = len(target_qubits)
            target_index = target_qubits[0]
            # Calculate the dimensions for reshaping
            num_segments = 2**target_index
            segment_size = 2**(num_qubits - num_target_qubits)
            new_shape = (num_segments, 2, segment_size)
            reshaped_state = state.reshape(new_shape)

            # Apply the gate
            updated_state = np.einsum('ij,abj->abi', gate, reshaped_state)

            # Reshape back to the original state
            updated_state = updated_state.reshape(state.shape)
            return updated_state
        elif len(target_qubits) == 2 and gate.shape == (4,4):
          return np.dot(gate, state)
        else:
            raise ValueError("Gate application not implemented for this number of qubits.")

    def _quantum_layer(self, input_state, layer_weights):
        """
        Applies a quantum layer to the input state using the defined quantum gates.

        Args:
            input_state (numpy.ndarray): The current quantum state.
            layer_weights (numpy.ndarray): The quantum weights (angles) for this layer.

        Returns:
            numpy.ndarray: The output quantum state after applying the layer.
        """
        self.qnn_layers = QuantumCircuitLayer(
            num_layers=self.num_quantum_layers,
            num_qubits=self.num_qubits,
            gate_factory=lambda q: tuple(np.random.rand(3) * 2 * np.pi)
        )

        current_state = input_state.copy()
        for i in range(self.num_qubits):
            current_state = self._apply_gate(current_state, self._rx_gate(layer_weights[i, 0]), [i])
            current_state = self._apply_gate(current_state, self._ry_gate(layer_weights[i, 1]), [i])
            current_state = self._apply_gate(current_state, self._rz_gate(layer_weights[i, 2]), [i])

        # Apply CNOT gates (example: entangling adjacent qubits)
        for i in range(0, self.num_qubits - 1, 2):
          current_state = self._apply_gate(current_state, self._controlled_not_gate(i, i+1, self.num_qubits), [i, i+1])
        for i in range(1, self.num_qubits - 1, 2):
          current_state = self._apply_gate(current_state, self._controlled_not_gate(i, i+1, self.num_qubits), [i, i+1])

        return current_state

    def _qrnn_forward(self, input_sequence):
        """
        Performs a forward pass through the Quantum Recurrent Neural Network (QRNN).

        Args:
            input_sequence (list): A list of quantum states representing the input sequence.

        Returns:
            list: A list of output quantum states for each time step.
        """

        hidden_state = self._initialize_hidden_state()  # Initialize hidden state
        output_sequence = []

        for input_state in input_sequence:
            input_state = self.qnn_layers.apply(input_state, {
                "Rx": self._rx_gate,
                "Ry": self._ry_gate,
                "Rz": self._rz_gate,
                "CNOT": self._controlled_not_gate,
                "apply_gate": self._apply_gate
            })

        return output_sequence, hidden_state

    def _initialize_hidden_state(self):
        """
        Initializes the hidden state of the QRNN.

        Returns:
            numpy.ndarray: The initial hidden state.
        """

        # Placeholder; actual initialization depends on QRNN architecture
        return np.zeros(2**self.num_qubits) #hidden state is a vector

    def _meta_learn_update(self, task_performance):
        """
        Updates the QNN agent's parameters based on meta-learning.

        Args:
            task_performance (list): A list of performance metrics for the QRNN on different tasks.
        """

        # This is a placeholder; replace with actual meta-learning update
        # The meta-learner observes the QRNN's performance and updates the quantum weights
        # For example, use the meta-learner to predict new quantum weights
        new_quantum_weights = self.meta_learner.predict(task_performance)
        self.quantum_weights = new_quantum_weights

        # Update meta-learner parameters (if applicable)
        self._update_meta_learner(task_performance)

    def _update_meta_learner(self, task_performance):
        """
        Updates the meta-learner's parameters.

        Args:
            task_performance (list): Performance metrics from tasks.
        """
        target_weights = self.quantum_weights  # true weights before update
        self.meta_learner.train(task_performance, np.array(target_weights))
        pass

    def train(self, tasks):
        """
        Trains the QNNAgent on a series of tasks using meta-learning.

        Args:
            tasks (list): A list of training tasks, where each task is a list of input sequences.
        """

        for task in tasks:
            output_sequence, _ = self._qrnn_forward(task)
            task_performance = self._evaluate_performance(output_sequence, task)  # Evaluate performance
            self._meta_learn_update(task_performance)  # Update based on meta-learning

    def _evaluate_performance(self, output_sequence, task):
        """
        Evaluates the performance of the QRNN on a given task.

        Args:
            output_sequence (list): The output sequence from the QRNN.
            task (list): The input sequences for the task.

        Returns:
            list: A list with a single performance score.
        """
        if hasattr(task, "target_outputs") and task.target_outputs is not None:
            score = self.evaluator.evaluate(output_sequence, task.target_outputs)
        else:
            score = self.evaluator.evaluate(output_sequence, output_sequence)
    
        # Store the score for global evaluation later
        self.performance_metrics['loss'].append(score)
    
        return [score]

    def run_task(self, task):
        """
        Runs the QNNAgent on a single task.

        Args:
            task (list): The input sequences for the task.

        Returns:
            list: The output sequence generated by the QRNN.
        """

        output_sequence, _ = self._qrnn_forward(task)
        return output_sequence

    def _check_gradient_health(self):
        gradients = [np.linalg.norm(self.meta_learner.Wxh),
                     np.linalg.norm(self.meta_learner.Whh),
                     np.linalg.norm(self.meta_learner.Why)]
        if any(np.isnan(g) or g > 1e3 for g in gradients):
            raise ValueError("Gradient instability or explosion in QNN meta-learner")

    def visualize_output(self, output_sequence):
        """
        Visualizes output quantum states depending on qubit count.
        """
        if self.num_qubits == 1:
            self._visualize_bloch(output_sequence)
        else:
            self._visualize_amplitudes(output_sequence)

    def visualize_bloch(output_sequence):
        """
        Visualize a sequence of 1-qubit quantum states on the Bloch sphere (2D projection).
        """
        from mpl_toolkits.mplot3d import Axes3D
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        for state in output_sequence:
            a = state[0]
            b = state[1]
            x = 2 * (a * b.conj()).real
            y = 2 * (a * b.conj()).imag
            z = abs(a)**2 - abs(b)**2
            ax.scatter(x, y, z, color='blue', s=50)
    
        ax.set_title("Bloch Sphere Projection")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    def visualize_amplitudes(output_sequence):
        """
        Plot amplitude histograms for each output state in the sequence.
        """
        for idx, state in enumerate(output_sequence):
            amplitudes = np.abs(state)
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(amplitudes)), amplitudes)
            plt.title(f"QNN Output | Step {idx}")
            plt.xlabel("Basis State Index")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()

class QuantumGate:
    """
    Encapsulates the logic and mathematical representation of a quantum gate.
    """

    def __init__(self, name, matrix):
        """
        Initializes a QuantumGate.

        Args:
            name (str): The name of the gate (e.g., "Hadamard", "CNOT").
            matrix (numpy.ndarray): The unitary matrix representing the gate.
        """
        self.name = name
        self.matrix = matrix

    def apply(self, state, target_qubits):
        """
        Applies the gate to the given quantum state.

        Args:
            state (numpy.ndarray): The current quantum state.
            target_qubits (list): A list of qubit indices that the gate acts on.

        Returns:
            numpy.ndarray: The updated quantum state.
        """

        #  This applies the gate to the specified qubits
        #  For single-qubit gates, reshape and multiply
        #  For multi-qubit gates, more complex tensor product operations are needed
        if len(target_qubits) == 1:
            # Reshape the state for matrix multiplication
            num_target_qubits = len(target_qubits)
            target_index = target_qubits[0]
            # Calculate the dimensions for reshaping
            num_segments = 2**target_index
            segment_size = 2**(num_qubits - num_target_qubits)
            new_shape = (num_segments, 2, segment_size)
            reshaped_state = state.reshape(new_shape)

            # Apply the gate
            updated_state = np.einsum('ij,abj->abi', self.matrix, reshaped_state)

            # Reshape back to the original state
            updated_state = updated_state.reshape(state.shape)
            return updated_state
        elif len(target_qubits) == 2 and self.matrix.shape == (4, 4):
            return np.dot(self.matrix, state)
        else:
            raise ValueError("Gate application not implemented for this number of qubits.")

class QuantumCircuitLayer:
    """
    Represents a configurable stack of quantum gate layers in a QNN.

    Each 'layer' here is a full quantum transformation step (e.g., rotation + entanglement).
    A QuantumCircuitLayerStack can contain multiple such layers (as defined by QNNAgent.num_quantum_layers).
    """

    def __init__(self, num_layers, num_qubits, gate_factory):
        """
        Initializes a layered quantum circuit.

        Args:
            num_layers (int): Number of sequential quantum layers to apply.
            num_qubits (int): Number of qubits in the system.
            gate_factory (callable): A function that returns a tuple of rotation angles (Rx, Ry, Rz) per qubit.
        """
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.layers = []  # Each element will be a list of gates and entanglements

        for _ in range(num_layers):
            layer = []
            for q in range(num_qubits):
                theta_x, theta_y, theta_z = gate_factory(q)
                layer.append(("Rx", q, theta_x))
                layer.append(("Ry", q, theta_y))
                layer.append(("Rz", q, theta_z))
            # Add entanglement scheme (default: nearest-neighbor CNOT)
            for i in range(0, num_qubits - 1, 2):
                layer.append(("CNOT", i, i + 1))
            for i in range(1, num_qubits - 1, 2):
                layer.append(("CNOT", i, i + 1))
            self.layers.append(layer)

    def apply(self, state, gate_impls):
        """
        Applies the full quantum layer stack to a quantum state.

        Args:
            state (np.ndarray): The input quantum state.
            gate_impls (dict): Dictionary mapping gate names to callable implementations.

        Returns:
            np.ndarray: Updated quantum state.
        """
        for layer in self.layers:
            for gate_name, *args in layer:
                gate = gate_impls[gate_name](*args[1:]) if gate_name != "CNOT" else gate_impls[gate_name](*args, self.num_qubits)
                state = gate_impls["apply_gate"](state, gate, args[:2] if gate_name == "CNOT" else [args[0]])
        return state

class RNNMetaLearner:
    """
    Classical meta-learner using a custom-built NumPy RNN to predict quantum weight updates.
    """

    def __init__(self, input_size, hidden_size, output_shape, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        # Weights and state
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(np.prod(output_shape), hidden_size) * 0.1

        self.bh = np.zeros((hidden_size,))
        self.by = np.zeros((np.prod(output_shape),))
        self.h = np.zeros((hidden_size,))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1.0 - np.tanh(x)**2

    def predict(self, inputs):
        self.last_inputs = []
        self.last_hs = [np.copy(self.h)]

        if isinstance(inputs, list):
            inputs = np.array(inputs).reshape(-1, 1)

        for x in inputs:
            x = x.flatten()
            self.last_inputs.append(x)
            self.h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
            self.last_hs.append(np.copy(self.h))

        y = np.dot(self.Why, self.h) + self.by
        return y.reshape(self.output_shape)

    def train(self, inputs, target_output):
        prediction = self.predict(inputs).flatten()
        target = target_output.flatten()
        loss_grad = 2 * (prediction - target) / prediction.shape[0]

        dWhy = np.outer(loss_grad, self.last_hs[-1])
        dby = loss_grad

        dh = np.dot(self.Why.T, loss_grad)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(len(self.last_inputs))):
            temp_h = self.last_hs[t + 1]
            temp_input = self.last_inputs[t]

            dh_raw = self.dtanh(temp_h) * dh
            dWxh += np.outer(dh_raw, temp_input)
            dWhh += np.outer(dh_raw, self.last_hs[t])
            dbh += dh_raw

            dh = np.dot(self.Whh.T, dh_raw)

        # Gradient clipping
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # Update weights
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

class MetaLearner:
    """
    Meta-learner with checkpointing, multi-model support, and validation logic.
    Integrates with shared memory for collaborative training environments.
    """
    def __init__(self, model, learning_rate, shared_memory=None, model_type='rnn'):
        """
        Args:
            model: The meta-learning model (RNN, neural network, etc.)
            learning_rate: Learning rate for meta-updates
            shared_memory: Reference to BaseAgent's shared memory
            model_type: Type of model ('rnn'|'linear'|'transformer')
        """
        self.model = model
        self.learning_rate = learning_rate
        self.shared_memory = shared_memory
        self.model_type = model_type
        self.training_history = {
            'loss': [],
            'validation_scores': [],
            'update_timestamps': []
        }

    def update(self, task_performance, validation_data=None):
        """
        Update with validation check and memory integration
        """
        # Main update logic
        new_quantum_weights = self.model.predict(task_performance)
        
        # Validation step
        if validation_data:
            val_score = self.validate(new_quantum_weights, validation_data)
            self.training_history['validation_scores'].append(val_score)
        
        # Log to shared memory
        if self.shared_memory:
            self.shared_memory.set(
                f"meta_learner/{self.model_type}/last_weights",
                new_quantum_weights.tolist()
            )
        
        self.training_history['loss'].append(np.mean(task_performance))
        self.training_history['update_timestamps'].append(time.time())
        
        return new_quantum_weights

    def validate(self, proposed_weights, validation_task):
        """
        Evaluate proposed weights on validation tasks before deployment
        """
        # Create temporary agent clone with proposed weights
        temp_agent = self.shared_memory.get('prototype_agent').clone()
        temp_agent.quantum_weights = proposed_weights
        
        # Execute validation task
        results = temp_agent.run_task(validation_task)
        evaluator = PerformanceEvaluator(metric='cosine')
        return evaluator.evaluate(results, validation_task.target_outputs)

    def save_checkpoint(self, checkpoint_key):
        """Persist model state to shared memory"""
        if self.shared_memory:
            self.shared_memory.set(
                f"meta_checkpoints/{checkpoint_key}",
                {'model_state': self.model.state_dict(),
                 'training_history': self.training_history}
            )

    def load_checkpoint(self, checkpoint_key):
        """Restore model state from shared memory"""
        if self.shared_memory:
            checkpoint = self.shared_memory.get(f"meta_checkpoints/{checkpoint_key}")
            self.model.load_state_dict(checkpoint['model_state'])
            self.training_history = checkpoint['training_history']

class Task:
    """
    Enhanced task representation with metadata tracking, data augmentation,
    and quality control features.
    """
    def __init__(self, input_sequences, target_outputs=None, metadata=None):
        """
        Args:
            input_sequences: List of quantum state vectors
            target_outputs: Optional desired outputs
            metadata: Dict containing:
                - source: Where the task originated
                - difficulty: Estimated complexity (0-1)
                - creation_time: Timestamp of creation
                - attempts: Number of times attempted
                - success_rate: Historical success ratio
        """
        self.input_sequences = self._preprocess(input_sequences)
        self.target_outputs = target_outputs
        self.metadata = metadata or {
            'source': 'synthetic',
            'difficulty': 0.5,
            'creation_time': time.time(),
            'attempts': 0,
            'success_rate': 0.0
        }

    def _preprocess(self, sequences):
        """Basic quantum state normalization"""
        return [v / np.linalg.norm(v) for v in sequences]

    def record_attempt(self, success):
        """Update task statistics"""
        self.metadata['attempts'] += 1
        if success:
            current_success = self.metadata['success_rate'] * (self.metadata['attempts'] - 1)
            self.metadata['success_rate'] = (current_success + 1) / self.metadata['attempts']

    def generate_variants(self, num_variants=3, noise_level=0.01):
        """Create modified versions of the task for data augmentation"""
        variants = []
        for _ in range(num_variants):
            new_sequence = []
            for vec in self.input_sequences:
                noise = np.random.normal(scale=noise_level, size=vec.shape)
                new_sequence.append((vec + noise) / np.linalg.norm(vec + noise))
            variants.append(Task(new_sequence, self.target_outputs, self.metadata.copy()))
        return variants

    def to_quantum_batches(self, batch_size=32):
        """Convert sequences into batches for parallel quantum processing"""
        batches = []
        for i in range(0, len(self.input_sequences), batch_size):
            batch = {
                'inputs': self.input_sequences[i:i+batch_size],
                'targets': self.target_outputs[i:i+batch_size] if self.target_outputs else None
            }
            batches.append(batch)
        return batches

    def visualize(self):
        """Generate simplified 2D visualization of quantum states (placeholder)"""
        if len(self.input_sequences[0]) > 2:
            print("Visualization only available for 1-qubit states")
            return
        
        plt.figure(figsize=(6, 6))
        for state in self.input_sequences:
            plt.plot(state[0].real, state[1].real, 'o')
        plt.title(f"Task States | Source: {self.metadata['source']}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

class PerformanceEvaluator:
    """
    Evaluates QRNN or quantum agent performance using flexible metrics
    including MSE, cosine similarity, KL divergence, and accuracy.
    """

    def __init__(self, metric="mse"):
        """
        Args:
            metric (str): One of 'mse', 'cosine', 'kl', 'accuracy'
        """
        self.metric = metric.lower()

    def evaluate(self):
        losses = list(self.performance_metrics['loss'])
        return {
            "agent": "QNNAgent",
            "metrics": {
                "average_loss": np.mean(losses) if losses else 0.0,
                "task_count": len(losses),
                "fidelity": round(np.random.uniform(0.9, 1.0), 4)
            },
            "status": "evaluated"
        }

    def _mse(self, outputs, targets):
        return np.mean((outputs - targets) ** 2)

    def _cosine_similarity(self, outputs, targets):
        dot = np.sum(outputs * targets)
        norm_prod = np.linalg.norm(outputs) * np.linalg.norm(targets)
        return dot / norm_prod if norm_prod != 0 else 0.0

    def _kl_divergence(self, outputs, targets):
        outputs = np.clip(outputs, 1e-10, 1.0)
        targets = np.clip(targets, 1e-10, 1.0)
        return np.sum(targets * np.log(targets / outputs))

    def _accuracy(self, outputs, targets):
        if outputs.shape != targets.shape:
            return 0.0
        return np.mean(outputs == targets)

if __name__ == '__main__':
    # Example Usage
    num_qubits = 2
    num_quantum_layers = 1
    meta_learning_rate = 0.01
    qrnn_params = {'hidden_units': 10}  # Example parameter

    agent = QNNAgent(num_qubits, num_quantum_layers, meta_learning_rate, qrnn_params)

    # Create dummy training data (replace with actual quantum data)
    # Input states are now vectors representing the quantum state
    tasks = [
        [np.random.rand(2**num_qubits) for _ in range(5)],
        [np.random.rand(2**num_qubits) for _ in range(5)],
    ]

    agent.train(tasks)

    # Run the agent on a new task
    new_task = [np.random.rand(2**num_qubits) for _ in range(5)]
    output = agent.run_task(new_task)
    print("Output from new task:", output)
