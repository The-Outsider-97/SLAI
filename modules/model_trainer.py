
# import resource
import os, sys
import joblib
import io
import gc
import time
import psutil
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from queue import Queue
from hashlib import sha256
from threading import Thread
from PyQt5.QtWidgets import QLabel
from collections import defaultdict
from PyQt5.QtCore import pyqtSignal, QObject
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# from modules.deployment.model_registry import register_model

from src.agents.collaborative.shared_memory import SharedMemory
from logs.logger import get_logger

logger = get_logger('SafeAI.ModelTrainer')

matplotlib.use("Qt5Agg")

class MonitorSignal(QObject):
    update_canvas = pyqtSignal(object)


class ModelTrainer:
    """Enhanced model trainer with memory-aware training and diagnostic capabilities"""
    
    def __init__(self, output_dir="models/", config=None):
        self.shared_memory = SharedMemory(config)
        self.output_dir = output_dir
        self._training_history = []
        self._feature_importance = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def train_model(self, data, model_type="logistic", params=None, 
                   test_size=0.2, random_state=42, cross_validate=False):
        """Enhanced training with memory monitoring and cross-validation"""
        X, y = data
        self._validate_data_shape(X, y)
        self._check_class_balance(y)
        
        # Memory baseline
        mem_before = self._get_memory_usage()
        start_time = time.time()

        try:
            model, metrics = self._train_model_internal(
                X, y, model_type, params, test_size, random_state, cross_validate
            )
        except MemoryError:
            logger.warning("Memory overflow detected, switching to fallback model")
            model, metrics = self._train_fallback_model(X, y, test_size)

        # Memory analysis
        training_time = time.time() - start_time
        mem_after = self._get_memory_usage()
        
        # Create model fingerprint
        metadata = {
            "training_time": training_time,
            "memory_delta": mem_after - mem_before,
            "data_hash": self._dataset_fingerprint(X, y),
            "metrics": metrics,
            "feature_importance": self._calculate_feature_importance(model, X.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        }

        model_path = self._persist_model(model, model_type, metadata)
        self._log_training_session(metadata)
        
        if self.shared_memory:
            self.shared_memory.set("model_metrics", metrics)
            self.shared_memory.set("trained_model_path", model_path)

        return model, metrics

    def _train_model_internal(self, X, y, model_type, params, test_size, random_state, cross_validate):
        """Core training logic with cross-validation support"""
        if cross_validate:
            return self._cross_validate(X, y, model_type, params, random_state)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = self._init_model(model_type, params)
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        
        return model, metrics

    def _init_model(self, model_type, params):
        """Model initialization with resource constraints"""
        model_config = {
            "logistic": {
                "class": LogisticRegression,
                "defaults": {"max_iter": 1000, "solver": 'lbfgs'}
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "defaults": {"n_estimators": 100}
            }
        }
        
        if model_type not in model_config:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_config[model_type]["class"](
            **{**model_config[model_type]["defaults"], **(params or {})}
        )

    def _cross_validate(self, X, y, model_type, params, random_state):
        """Manual k-fold cross-validation implementation"""
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        fold_metrics = defaultdict(list)
        models = []

        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self._init_model(model_type, params)
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            
            for k, v in metrics.items():
                if isinstance(v, float):
                    fold_metrics[k].append(v)
            models.append(model)

        # Aggregate metrics
        avg_metrics = {f"avg_{k}": np.mean(v) for k, v in fold_metrics.items()}
        best_idx = np.argmax([m['accuracy'] for m in fold_metrics])
        best_model = models[best_idx]
        
        return best_model, avg_metrics

    def evaluate_model(self, model, X_test, y_test):
        """Enhanced evaluation with class-wise metrics"""
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average='weighted'), 4),
            "recall": round(recall_score(y_test, preds, average='weighted'), 4),
            "f1_score": round(f1_score(y_test, preds, average='weighted'), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4) if proba is not None else "N/A",
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
        
        if self.shared_memory:
            self.shared_memory.set("last_evaluation", metrics)
            
        return metrics

    def _persist_model(self, model, model_type, metadata):
        """Save model with metadata using joblib"""
        model_path = os.path.join(self.output_dir, f"{model_type}_{int(time.time())}.joblib")
        joblib.dump({
            "model": model,
            "metadata": metadata
        }, model_path)
        logger.info(f"Model saved with metadata to: {model_path}")
        return model_path

    def _get_memory_usage(self):
        """Cross-platform memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def _dataset_fingerprint(self, X, y):
        """Create dataset hash fingerprint for versioning"""
        data_hash = sha256(pd.util.hash_pandas_object(X).values).hexdigest()
        target_hash = sha256(pd.util.hash_pandas_object(y).values).hexdigest()
        return f"{data_hash[:8]}_{target_hash[:8]}"

    def _check_class_balance(self, y):
        """Warn on significant class imbalance"""
        class_counts = y.value_counts(normalize=True)
        if np.min(class_counts) < 0.1:
            logger.warning(f"Class imbalance detected: {class_counts.to_dict()}")

    def _validate_data_shape(self, X, y):
        """Verify data integrity before training"""
        if len(X) != len(y):
            raise ValueError("Feature/target length mismatch")
        if X.isna().any().any():
            raise ValueError("NaN values detected in features")
        if y.isna().any():
            raise ValueError("NaN values detected in target")

    def _train_fallback_model(self, X, y, test_size):
        """Simplified model for resource-constrained environments"""
        logger.info("Training fallback logistic model")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        return model, self.evaluate_model(model, X_test, y_test)

    def _calculate_feature_importance(self, model, feature_names):
        """Extract feature importance based on model type"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
        return {}

    def load_model(self, model_path):
        """Load model with metadata verification"""
        artifact = joblib.load(model_path)
        logger.info(f"Loaded model trained at {artifact['metadata']['timestamp']}")
        return artifact['model']

    def release_resources(self):
        """Explicit resource cleanup"""
        gc.collect()
        logger.info("Memory resources released")

    def _log_training_session(self, metadata):
        """Maintain training history with diagnostic data"""
        self._training_history.append(metadata)
        logger.info(f"Training session recorded: {metadata['data_hash']}")

class ModelTrainingPipeline:
    def __init__(self, output_dir="models/", dashboard_widget=None):
        self.trainer = ModelTrainer(output_dir, config=None)
        self.task_queue = Queue()
        self.monitor_data = []
        self.monitoring = True
        self.dashboard_widget = dashboard_widget
        self.monitor_signal = MonitorSignal()
        self.monitor_signal.update_canvas.connect(self._render_dashboard_plot)
        self.monitor_thread = Thread(target=self._monitor_memory_usage, daemon=True)

    def run(self, dataset_path, target_column, model_type="logistic", params=None, cross_validate=False):
        self.monitor_thread.start()
        data = self._load_and_preprocess(dataset_path, target_column)
        self.task_queue.put((data, model_type, params, cross_validate))
        model, metrics = self._process_tasks()
        self.monitoring = False
        self.trainer.release_resources()
        return model, metrics

    def _load_and_preprocess(self, path, target_column):
        df = pd.read_csv(path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def _process_tasks(self):
        model, metrics = None, None
        while not self.task_queue.empty():
            data, model_type, params, cross_validate = self.task_queue.get()
            model, metrics = self.trainer.train_model(
                data,
                model_type=model_type,
                params=params,
                cross_validate=cross_validate
            )
            self.task_queue.task_done()
        return model, metrics

    def _monitor_memory_usage(self):
        while self.monitoring:
            mem = psutil.Process().memory_info().rss / (1024 ** 2)  # in MB
            self.monitor_data.append((time.time(), mem))
            time.sleep(0.5)
            if self.dashboard_widget:
                self.monitor_signal.update_canvas.emit(self.monitor_data[:])

    def _render_dashboard_plot(self, monitor_data):
        if not monitor_data:
            return
        timestamps, mem_usage = zip(*monitor_data)
        timestamps = [t - timestamps[0] for t in timestamps]
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(timestamps, mem_usage, label="Memory Usage (MB)", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Live Memory Monitoring")
        ax.grid(True)
        canvas = FigureCanvas(fig)

        # Clear and embed into dashboard widget
        for i in reversed(range(self.dashboard_widget.layout().count())):
            self.dashboard_widget.layout().itemAt(i).widget().setParent(None)
        self.dashboard_widget.layout().addWidget(canvas)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Model Trainer ===\n")
    output_dir = "models/"
    config = None

    trainer = ModelTrainer(output_dir=output_dir, config=config)
    logger.info(f"{trainer}")

    import glob
    model_files = glob.glob(os.path.join(output_dir, "*.joblib"))
    if not model_files:
        print("No model files found. Exiting.")
        sys.exit(1)

    latest_model = max(model_files, key=os.path.getctime)
    model = trainer.load_model(model_path=latest_model)
    print(f"\n* * * * * Phase 2 * * * * *\n")
    output_dir="modules/"
    dashboard_widget=None

    pipeline = ModelTrainingPipeline(output_dir=output_dir, dashboard_widget=dashboard_widget)
    logger.info(f"{pipeline}")
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Model Trainer ===\n")
