
from PyQt5.QtCore import QObject, pyqtSignal, Qt

class TrainingSignals(QObject):
    """
    Central signal hub for coordinating UI and worker thread communication.
    All signals are explicitly designed for Qt.QueuedConnection usage to ensure
    thread safety when interacting with QWidget or other GUI components.
    """
    
    # Signals for text logs or status updates
    log_message = pyqtSignal(str)  # Text messages for log panels
    batch_status = pyqtSignal(str)  # Status updates, e.g., APPROVED, REJECTED
    
    # Signals for embedding updates (from background processing)
    embedding_update = pyqtSignal(dict)  # Send {word, embedding, decision, ...}
    
    # Signals for approval and decisions in the synonym trainer
    approval_request = pyqtSignal(str, dict, str)  # word, entry block, entry type
    approval_result = pyqtSignal(str, bool)  # word, approval (True/False)
    
    # Signals for controlling batch progression
    batch_continuation = pyqtSignal(int, int)  # processed count, remaining count
    batch_approved = pyqtSignal(bool)  # Whether the batch was approved
    batch_rejected = pyqtSignal(bool)  # Whether the batch was rejected

    def __init__(self):
        super().__init__()

# Global shared instance (singleton)
training_signals = TrainingSignals()

def safe_connect(signal, target_slot):
    """
    Safely connects a signal to a slot using Qt.QueuedConnection.
    This guarantees the slot executes in the main (GUI) thread even
    when the signal is emitted from a worker or non-GUI thread.
    
    Args:
        signal (pyqtSignal): The signal to connect.
        target_slot (callable): The function or method to connect to.
    """
    try:
        signal.connect(target_slot, Qt.QueuedConnection)
    except Exception as e:
        print(f"[ERROR] Failed to connect signal {signal} to slot {target_slot}: {e}")
