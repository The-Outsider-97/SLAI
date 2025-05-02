
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QTimer
def safe_connect(signal, target_slot):
    signal.connect(lambda *args, **kwargs:
        QTimer.singleShot(0, lambda: target_slot(*args, **kwargs))
    )

class TrainingSignals(QObject):
    # Synonym Trainer Signals
    approval_request = pyqtSignal(str, dict, str)  # word, entry, entry_type
    log_message = pyqtSignal(str)
    batch_continuation = pyqtSignal(int, int)  # processed, remaining
    batch_approved = pyqtSignal(bool)
    batch_rejected = pyqtSignal(bool)
    batch_status = pyqtSignal(str)
    
    # Embedding Signals
    embedding_update = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.embedding_data = {}  # Shared embedding storage

training_signals = TrainingSignals()
