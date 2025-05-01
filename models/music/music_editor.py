import time
import queue
import threading
import numpy as np
import sounddevice as sd  # lightweight, pure-Python audio I/O

from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout,QHBoxLayout,
    QSlider, QFrame, QGridLayout, QSizePolicy,)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QPropertyAnimation, QSize, QRect
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QIcon
from collections import deque
from threading import Lock
from typing import Dict

from src.agents.perception_agent import PerceptionAgent


class AudioStreamer:
    def __init__(self, buffer_size=16000, samplerate=44100, channels=1, blocksize=1024, callback=None):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.callback = callback
        self.processing_thread = threading.Thread(target=self.start_processing, daemon=True)
        self.stream = None
        self.running = False
        self.lock = Lock()

        self.audio_buffer = np.zeros((blocksize,), dtype=np.float32)
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.write_idx = 0

        self.queue = queue.Queue(maxsize=100)

        self.restart_attempts = 0
        self.max_restart_attempts = 20  # Avoid infinite loops
        self.needs_restart = False

    def _audio_callback(self, indata, frames, time_info, status):
        #if self.audio_disabled:
        #    return
        if status:
            print(f"[AudioStreamer Warning] {status}")

            # 🛠 Detect if the stream is broken, try to restart
            if status.input_overflow:
                print("[AudioStreamer] Input Overflow detected, attempting restart...")
                #self._restart_stream()
                self.needs_restart = True

        # Always update audio buffer
        self.audio_buffer = indata[:, 0]  # Assuming mono input

        # Handle incoming samples
        new_samples = indata[:, 0]
        n = len(new_samples)
        available = len(self.buffer) - self.write_idx

        if n > available:
            self.buffer[self.write_idx:] = new_samples[:available]
            self.buffer[:n - available] = new_samples[available:]
            self.write_idx = n - available
        else:
            self.buffer[self.write_idx:self.write_idx + n] = new_samples
            self.write_idx += n

        if self.callback:
            try:
                self.queue.put_nowait(indata.copy())
            except queue.Full:
                print("[AudioStreamer] Queue full, dropping old frame.")

    def _restart_stream(self):
        if self.restart_attempts >= self.max_restart_attempts:
            print("[AudioStreamer] Maximum restart attempts reached. Disabling microphone input.")
            self.audio_disabled = True
            return

        self.restart_attempts += 1
        print(f"[AudioStreamer] Restart attempt {self.restart_attempts}...")

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"[AudioStreamer] Error stopping old stream: {e}")

        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype='float32',
                callback=self._audio_callback,
            )
            self.stream.start()
            self.running = True
            self.restart_attempts = 0
            print("[AudioStreamer] Stream restarted successfully.")
            self.restart_attempts = 0  # Reset counter on success
        except Exception as e:
            print(f"[AudioStreamer] Stream restart failed: {e}")

    def start_processing(self):
        while True:
            data = self.queue.get()
            if self.channels == 1:
                data = data.flatten()
            self.callback(data)

    def start(self):
        if self.running:
            print("[AudioStreamer] Already running.")
            return
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.running = True
            print("[AudioStreamer] Started.")
            # Start background thread if not already running
            if not self.processing_thread.is_alive():
                self.processing_thread.start()
        except Exception as e:
            print(f"[AudioStreamer] Failed to start: {e}")
            self.running = False

    def stop(self):
        if not self.running:
            print("[AudioStreamer] Already stopped.")
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            print(f"[AudioStreamer] Error while stopping: {e}")
        self.running = False
        print("[AudioStreamer] Stopped.")

    def get_buffer(self):
        with self.lock:
            return np.copy(self.audio_buffer)

class WaveformWidget(QWidget):
    class PerceptionWorker(QThread):
        finished = pyqtSignal(np.ndarray)

        def __init__(self, agent, buffer):
            super().__init__()
            self.agent = agent
            self.buffer = buffer.copy()
            self._interrupted = False
            self.setTerminationEnabled(True)

        def run(self):
            if self._interrupted:
                return
            try:
                samples = self.buffer.reshape(1, 1, -1)  # batch, channels, samples
                audio_input = {"audio": samples}
                embedding = self.agent.forward(audio_input)
                if not self._interrupted:
                    self.finished.emit(embedding)
            except Exception as e:
                print(f"[PerceptionWorker] Error: {str(e)}")

        def interrupt(self):
            self._interrupted = True

    def __init__(self, audio_streamer, perception_agent=None, parent=None, max_embeddings=100):
        super().__init__(parent)
        self.audio_streamer = audio_streamer
        self.perception_agent = perception_agent
        self.frame_buffer = np.zeros(16000, dtype=np.float32)
        self.frame_write_idx = 0
        self.active_worker = None
        self.worker_lock = Lock()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)  # ~33 FPS

		# Embedding Memory
        self.embedding_memory: Dict[str, np.ndarray] = {}
        self.max_embeddings = max_embeddings
        self.embedding_keys = deque()  # To maintain order for eviction

        self.hanning_window = None  # Will be initialized based on buffer size
        self.volume_width = 4
        self.freq_width = 8  
        self.note_width = 12

    def _tick(self):
        self.update()

    def process_buffer(self, buffer):
        """Ensure buffer is exactly 16000 samples"""
        if len(buffer) < 16000:
            padded = np.zeros(16000, dtype=np.float32)
            padded[: len(buffer)] = buffer
            return padded
        elif len(buffer) > 16000:
            return buffer[:16000]
        return buffer

    def frequency_to_note(self, freq):
        """Convert frequency to note name and cents deviation."""
        if freq <= 0:
            return ("", 0.0)
        try:
            note_num = 12 * np.log2(freq / 440.0) + 69
            note_num_rounded = int(round(note_num))
            cents = (note_num - note_num_rounded) * 100
            note_num_rounded = max(0, min(127, note_num_rounded))
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = (note_num_rounded // 12) - 1
            note_index = note_num_rounded % 12
            return (f"{notes[note_index]}{octave}", cents)
        except:
            return ("", 0.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        pen = QPen(QColor("gold"))
        pen.setWidth(2)
        painter.setPen(pen)

        buffer = self.audio_streamer.get_buffer()
        if len(buffer) == 0:
            painter.drawText(self.rect(), Qt.AlignCenter, "No audio input")
            return

        w = self.width()
        h = self.height()
        mid = h // 2
        step = max(1, len(buffer) // w)

        for x in range(w):
            idx = min(x * step, len(buffer) - 1)
            sample = np.clip(buffer[idx], -1.0, 1.0)
            y = int(mid - sample * mid)
            painter.drawLine(x, mid, x, y)

        # Calculate metrics
        rms = np.sqrt(np.mean(buffer**2))
        volume_value = int(np.clip(rms * 300, 0, 300))
        
        N = len(buffer)
        if N < 2:
            frequency = 0.0
            note_name, cents = "", 0.0
        else:
            if self.hanning_window is None or len(self.hanning_window) != N:
                self.hanning_window = np.hanning(N)
            windowed = buffer * self.hanning_window
            fft_result = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(N, d=1.0/self.audio_streamer.samplerate)
            magnitudes = np.abs(fft_result)
            if len(magnitudes) > 1:
                peak_index = np.argmax(magnitudes[1:]) + 1
                frequency = freqs[peak_index]
            else:
                frequency = 0.0
            note_name, cents = self.frequency_to_note(frequency)

        # Fixed width formatting
        info_text = (
            f"{volume_value:>{self.volume_width}d}dB | "
            f"{frequency:>{self.freq_width-2}.1f}Hz | "
            f"{note_name} {cents:+6.1f}¢".ljust(self.note_width)
        )

        # Draw metrics text
        painter.setPen(QColor("gold"))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        #painter.drawText(10, 20, info_text)

        self._collect_frame(buffer)

        # Define box dimensions
        box_width = 120
        box_height = 40
        spacing = 20
        start_y = 10
        
        # Create three rectangles for metrics
        volume_rect = QRect(10, start_y, box_width, box_height)
        freq_rect = QRect(10, start_y + box_height + spacing, box_width, box_height)
        note_rect = QRect(10, start_y + 2*(box_height + spacing), box_width, box_height)

        # Style parameters
        box_style = {
            'background': QColor(14, 14, 14),  # Dark background
            'border': QPen(QColor("gold"), 2),
            'text': QPen(QColor("gold")),
            'font': QFont('Monospace', 10, QFont.Bold)
        }

        def draw_metric_text(painter, pos_x, pos_y, value):
            painter.setPen(QColor("gold"))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(pos_x, pos_y, value)

        # Metrics
        start_x = 10
        start_y = 30
        spacing_x = 140

        volume_text = f"{volume_value:>3d} dB"
        freq_text = f"{frequency:>7.1f} Hz"
        note_text = f"{note_name}{cents:+6.1f}¢"

        draw_metric_text(painter, start_x + spacing_x * 0, start_y, volume_text)
        draw_metric_text(painter, start_x + spacing_x * 1, start_y, freq_text)
        draw_metric_text(painter, start_x + spacing_x * 2, start_y, note_text)

        self._collect_frame(buffer)

    def _collect_frame(self, buffer):
        """Collects audio data into full 16000-sample frames."""
        n = len(buffer)
        space_left = 16000 - self.frame_write_idx

        if n >= space_left:
            self.frame_buffer[self.frame_write_idx :] = buffer[:space_left]
            complete_frame = self.process_buffer(self.frame_buffer)
            self._start_perception_worker(complete_frame)
            self.frame_write_idx = 0
        else:
            self.frame_buffer[
                self.frame_write_idx : self.frame_write_idx + n
            ] = buffer
            self.frame_write_idx += n

    def _collect_frame(self, buffer):
        """Collects audio data into full 16000-sample frames."""
        n = len(buffer)
        space_left = 16000 - self.frame_write_idx
        if n >= space_left:
            self.frame_buffer[self.frame_write_idx :] = buffer[:space_left]
            complete_frame = self.process_buffer(self.frame_buffer)
            self._start_perception_worker(complete_frame)
            self.frame_write_idx = 0
        else:
            self.frame_buffer[
                self.frame_write_idx : self.frame_write_idx + n
            ] = buffer
            self.frame_write_idx += n

    def _start_perception_worker(self, buffer):
        """Launches a worker thread to process a full frame"""
        with self.worker_lock:
            if self.active_worker and self.active_worker.isRunning():
                self.active_worker.interrupt()
                self.active_worker.wait()  # Wait for termination
            self.active_worker = self.PerceptionWorker(self.perception_agent, buffer)
            self.active_worker.finished.connect(self._store_embedding)  # Connect the signal
            self.active_worker.start()

    def _store_embedding(self, embedding: np.ndarray):
        """Stores the generated embedding in memory."""
        embedding_key = (
            f"audio_frame_{time.time()}"  # Generate a unique key (e.g., timestamp)
        )
        if len(self.embedding_memory) >= self.max_embeddings:
            oldest_key = self.embedding_keys.popleft()
            del self.embedding_memory[oldest_key]  # Evict oldest
        self.embedding_memory[embedding_key] = embedding
        self.embedding_keys.append(embedding_key)
        print(
            f"[WaveformWidget] Stored embedding with key: {embedding_key} "
            f"(Memory size: {len(self.embedding_memory)})"
        )

    def get_embedding(self, key: str) -> np.ndarray:
        """Retrieves an embedding from memory."""
        return self.embedding_memory.get(key)

    def clear_embedding_memory(self):
        """Clears the embedding memory."""
        self.embedding_memory.clear()
        self.embedding_keys.clear()
        print("[WaveformWidget] Embedding memory cleared.")

    def _worker_done(self):
        """Called when the worker thread finishes."""
        with self.worker_lock:
            self.active_worker = None

    def handle_embedding_result(self, embedding):
        print(f"[WaveformWidget] Got live embedding shape: {embedding.shape}")
        parent_editor = self.parent()
        if parent_editor and hasattr(parent_editor, "store_embedding"):
            parent_editor.store_embedding(embedding)

    def closeEvent(self, event):
        """Ensure clean shutdown"""
        print("[WaveformWidget] Close event triggered.")
        self.timer.stop()

        if self.active_worker:
            print("[WaveformWidget] Interrupting active worker...")
            self.active_worker.interrupt()
            if not self.active_worker.wait(2000):  # Wait up to 2 seconds
                print(
                    "[WaveformWidget] Worker did not stop in time, terminating."
                )
                self.active_worker.terminate()
                self.active_worker.wait()

        event.accept()


class MusicEditor(QWidget):
    generateRequested = pyqtSignal()

    cutActionTriggered = pyqtSignal(int)
    copyActionTriggered = pyqtSignal(int)
    pasteActionTriggered = pyqtSignal(int)
    trimActionTriggered = pyqtSignal(int)
    zoominActionTriggered = pyqtSignal(int)
    zoomoutActionTriggered = pyqtSignal(int)
    fadeinActionTriggered = pyqtSignal(int)
    fadeoutActionTriggered = pyqtSignal(int)
    normalizeActionTriggered = pyqtSignal(int)

    keyChanged = pyqtSignal(int)
    tempoChanged = pyqtSignal(int)
    noiseChanged = pyqtSignal(int)
    reverbChanged = pyqtSignal(int)
    delayChanged = pyqtSignal(int)
    distortionChanged = pyqtSignal(int)
    pitchshiftChanged = pyqtSignal(int)
    volumeChanged = pyqtSignal(int)

    timeSignatureChanged = pyqtSignal(str)
    styleChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Store references to important widgets
        self.action_buttons = {}
        self.setting_sliders = {}
        self.edit_buttons = {}
        self.control_buttons = {}
        self.animations = []
        for btn in self.control_buttons.values():
            animation = QPropertyAnimation(btn, b"windowOpacity")
            animation.setDuration(800)
            animation.setStartValue(0)
            animation.setEndValue(1)
            animation.start()
            self.animations.append(animation)

        self.perception_agent = PerceptionAgent(
            config={
                "modalities": ["audio"],
                "embed_dim": 100,
                "projection_dim": 256,
            },
            shared_memory=None,
            agent_factory=None,
        )

        self.init_ui()
        self._connect_signals()  # Connect internal signals after UI is built
        QTimer.singleShot(1000, self.start_audio_streamer)

    def start_audio_streamer(self):
        """Start AudioStreamer safely after UI initialization."""
        if not self.audio_streamer.running:
            self.audio_streamer.start()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        self.audio_streamer = AudioStreamer(samplerate=16000, blocksize=1024)

        # --- Waveform + Edit Buttons Section ---
        waveform_frame = QFrame()
        waveform_frame.setStyleSheet(
            "background-color: #0e0e0e; border: 1px solid gold; border-radius: 8px;"
        )
        waveform_frame.setFixedHeight(300)
        wf_layout = QVBoxLayout(waveform_frame)
        wf_layout.setSpacing(10)
        self.waveform_widget = WaveformWidget(
            self.audio_streamer, self.perception_agent
        )
        wf_layout.addWidget(self.waveform_widget)

        # Editing Buttons
        edit_layout = QHBoxLayout()
        edit_button_names = [
            "Cut", "Copy", "Paste", "Trim", "Zoom In", "Zoom Out",
            "Fade In", "Fade Out", "Normalize",
        ]
        for name in edit_button_names:
            btn = QPushButton(name)
            btn.setStyleSheet(
                "background-color: gold; color: black; font-weight: bold; border-radius: 6px;"
            )
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edit_layout.addWidget(btn)
            self.edit_buttons[name] = btn  # <<< Store reference
        wf_layout.addLayout(edit_layout)
        layout.addWidget(waveform_frame)

        # --- Tab Bar ---
        tab_bar_layout = QHBoxLayout()
        tab_bar_layout.setSpacing(40)  # Space evenly
        tab_bar_layout.setContentsMargins(0, 1, 0, 1)
        tab_bar_layout.setAlignment(Qt.AlignCenter)

        self.tabs = {}
        for label in ["Player", "Composer", "Library"]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    color: #f9f9f9;
                    background: transparent;
                    border: none;
                    font-size: 16px;
                }
                QPushButton:checked {
                    color: #38b4fe;
                    border-bottom: 2px solid #38b4fe;
                }
            """)
            btn.clicked.connect(lambda _, b=label: self.switch_tab(b))
            self.tabs[label] = btn
            tab_bar_layout.addWidget(btn)
        
        layout.addLayout(tab_bar_layout)

        # --- Combined Action Buttons + Song Info Section ---
        action_info_frame = QFrame()
        action_info_frame.setStyleSheet(
            "background-color: #0e0e0e; border: 1px solid gold; border-radius: 8px;"
        )
        action_info_frame.setFixedHeight(250)
        action_info_layout = QVBoxLayout(action_info_frame)
        action_info_layout.setSpacing(10)

        # Action Buttons Grid
        actions_layout = QGridLayout()
        actions = ["Generate", "Lyrics", "Upscale", "Modify", "Mix", "Export", "Save", "Load"]
        for i, label in enumerate(actions):
            action_btn = QPushButton(label)
            action_btn.setStyleSheet(
                "background-color: gold; color: black; font-weight: bold; border: none; border-radius: 6px;"
            )
            action_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            actions_layout.addWidget(action_btn, 0, i)
            self.action_buttons[label] = action_btn
        action_info_layout.addLayout(actions_layout)

        # Song Info Labels
        song_info_layout = QVBoxLayout()
        info_labels = ["Title", "Artist", "Album", "Genre", "Release Year", "BPM", "Key", "Note"]
        for text in info_labels:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: white; font-weight: bold; padding-left: 4px;")
            song_info_layout.addWidget(lbl)
        action_info_layout.addLayout(song_info_layout)

        layout.addWidget(action_info_frame)

        # --- Fade-in Animation for the whole frame ---
        fade_animation = QPropertyAnimation(action_info_frame, b"windowOpacity")
        fade_animation.setDuration(1000)
        fade_animation.setStartValue(0)
        fade_animation.setEndValue(1)
        fade_animation.start()
        self.animations.append(fade_animation)

        # --- Song Settings Section ---
        settings_frame = QFrame()
        settings_frame.setStyleSheet(
            "background-color: transparent; "  # "border: 1px solid gold; border-radius: 6px;"
        )
        settings_layout = QVBoxLayout(settings_frame)
        settings_label = QLabel("Song Settings")
        settings_label.setStyleSheet(
            "color: gold; font-weight: bold; font-size: 18px;"
        )
        settings_layout.addWidget(settings_label)

        setting_sliders_config = [  # Renamed for clarity
            ("Volume", 0, 100),
            ("Key", 0, 11),
            ("Tempo", 40, 300),
            ("Noise", 0, 300),
            ("Reverb", 0, 100),
            ("Delay", 0, 100),
            ("Distortion", 0, 100),
            ("Pitch Shift", -12, 12),
        ]

        for name, min_val, max_val in setting_sliders_config:
            label = QLabel(f"{name}")
            label.setStyleSheet("color: gold; font-weight: bold;")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue((min_val + max_val) // 2)
            slider.setStyleSheet(
                "QSlider::groove:horizontal { "
                "background: gold; height: 6px; } "
                "QSlider::handle:horizontal { "
                "background: white; "
                "border: 1px solid gold; width: 14px; margin: -5px 0; border-radius: 7px; "
                "}"
            )
            settings_layout.addWidget(label)
            settings_layout.addWidget(slider)
            self.setting_sliders[name] = slider  # <<< Store reference
        layout.addWidget(settings_frame)

        # --- Bottom Music Controls (assuming these might also trigger actions) ---
        controls_layout = QHBoxLayout()
        control_buttons = [
            ("Shuffle", "frontend/assets/icons/shuffle.svg", "frontend/assets/icons/shuffle1.svg"),
            ("Previous", "frontend/assets/icons/previous.svg", "frontend/assets/icons/previous1.svg"),
            ("Play", "frontend/assets/icons/play.svg", "frontend/assets/icons/play1.svg"),
            ("Stop", "frontend/assets/icons/stop.svg", "frontend/assets/icons/stop1.svg"),
            ("Next", "frontend/assets/icons/next.svg", "frontend/assets/icons/next1.svg"),
            ("repeat", "frontend/assets/icons/repeat.svg", "frontend/assets/icons/repeat1.svg"),
        ]
        self.control_buttons = {}
        for label, gold_icon_path, white_icon_path in control_buttons:
            btn = ControlButton(gold_icon_path, white_icon_path)
            controls_layout.addWidget(btn)
            self.control_buttons[label] = btn
        layout.addLayout(controls_layout)

        self.setLayout(layout)
        self.setStyleSheet("background-color: black;")

    def switch_tab(self, selected):
        for label, button in self.tabs.items():
            button.setChecked(label == selected)
        # Handle actual content switching here (e.g., stackedWidget.setCurrentIndex)

    def _connect_signals(self):
        """Connect internal widget signals to the class's custom signals."""
        # Connect the 'Generate' button's clicked signal to our custom signal
        if "Generate" in self.action_buttons:
            self.action_buttons["Generate"].clicked.connect(
                self.generateRequested.emit
            )

        # Add connections for other buttons/sliders here...
        # Waveform Edit Buttons:
        if "Cut" in self.edit_buttons:
            self.edit_buttons["Cut"].clicked.connect(self.cutActionTriggered.emit)
        if "Copy" in self.edit_buttons:
            self.edit_buttons["Copy"].clicked.connect(self.copyActionTriggered.emit)
        if "Paste" in self.edit_buttons:
            self.edit_buttons["Paste"].clicked.connect(self.pasteActionTriggered.emit)
        if "Trim" in self.edit_buttons:
            self.edit_buttons["Trim"].clicked.connect(self.trimActionTriggered.emit)
        if "Zoom In" in self.edit_buttons:
            self.edit_buttons["Zoom In"].clicked.connect(self.zoominActionTriggered.emit)
        if "Zoom Out" in self.edit_buttons:
            self.edit_buttons["Zoom Out"].clicked.connect(self.zoomoutActionTriggered.emit)
        if "Fade In" in self.edit_buttons:
            self.edit_buttons["Fade In"].clicked.connect(self.fadeinActionTriggered.emit)
        if "Fade Out" in self.edit_buttons:
            self.edit_buttons["Fade Out"].clicked.connect(self.fadeoutActionTriggered.emit)
        if "Normalize" in self.edit_buttons:
            self.edit_buttons["Normalize"].clicked.connect(self.normalizeActionTriggered.emit)

        if "Volume" in self.setting_sliders:
            self.setting_sliders["Volume"].valueChanged.connect(self.volumeChanged.emit)
        if "Key" in self.setting_sliders:
            self.setting_sliders["Key"].valueChanged.connect(self.keyChanged.emit)
        if "Tempo" in self.setting_sliders:
            self.setting_sliders["Tempo"].valueChanged.connect(self.tempoChanged.emit)
        if "Noise" in self.setting_sliders:
            self.setting_sliders["Noise"].valueChanged.connect(self.noiseChanged.emit)
        if "Reverb" in self.setting_sliders:
            self.setting_sliders["Reverb"].valueChanged.connect(self.reverbChanged.emit)
        if "Delay" in self.setting_sliders:
            self.setting_sliders["Delay"].valueChanged.connect(self.delayChanged.emit)
        if "Distortion" in self.setting_sliders:
            self.setting_sliders["Distortion"].valueChanged.connect(self.distortionChanged.emit)
        if "Pitch Shift" in self.setting_sliders:
            self.setting_sliders["Pitch Shift"].valueChanged.connect(self.pitchshiftChanged.emit)

    def init_audio_engine(self, sr=44100, bit_depth=16, channels=1):
        # Share audio config with trainer
        self.sr = sr
        self.bit_depth = bit_depth
        self.channels = channels
        self.block_size = 1024  # samples per buffer/frame
        self.input_device_index = None  # or detect default mic
        self.output_device_index = None  # or detect default speaker
        self.stream = None

        self.log(f"[Audio Engine] Initialized with {self.sr} Hz, {self.bit_depth}-bit, {self.channels} channel(s).")

    def store_embedding(self, embedding):
        """Store live audio embeddings into memory safely."""
        if not hasattr(self, "embeddings"):
            self.embeddings = deque(maxlen=500)  # auto-delete old ones
        self.embeddings.append(embedding)
        # Validate embedding
        if embedding is None:
            print("[MusicEditor] Warning: Received None embedding.")
            return

        if not isinstance(embedding, np.ndarray):
            print("[MusicEditor] Warning: Received invalid embedding type:",
                type(embedding),)
            return

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print("[MusicEditor] Warning: Embedding contains NaN or Inf. Skipping.")
            return

        if embedding.ndim != 2:
            print(f"[MusicEditor] Warning: Expected 2D embedding, got {embedding.ndim}D. Skipping.")
            return

        if embedding.shape[1] != 256:
            print(f"[MusicEditor] Warning: Expected embedding_dim=256, got {embedding.shape[1]}. Skipping.")
            return

        self.embedding_memory.append(embedding)

        if len(self.embedding_memory) >= 32:
            self.train_musician_on_live_audio()

    def train_musician_on_live_audio(self):
        """Train Musician model using recent embeddings."""
        batch = np.stack(list(self.embedding_memory)[-32:], axis=0)  # (batch_size, embedding_dim)
        self.embedding_memory.clear()

        # Assume Musician expects a simple batch for training
        loss = self.musician.train_on_batch(batch)
        print(f"[MusicEditor] Trained Musician model, batch_loss={loss:.4f}")

    def process_audio_frame(self, frame):
        """
        Handle incoming audio frames from AudioStreamer in real-time.
        frame: numpy array (samples,)
        """
        # Simple version: just forward to PerceptionAgent every frame
        samples = frame.reshape(1, -1)  # (batch=1, samples)
        samples = samples[:, np.newaxis, :]  # (batch=1, channels=1, samples)

        audio_input = {"audio": samples}
        embedding = self.perception_agent.forward(audio_input)
        print(
            f"[MusicEditor] Processed live audio frame -> embedding shape: {embedding.shape}"
        )

    def _audio_callback(self, indata, frames, time_info, status):
        new_samples = indata[:, 0]
        n = len(new_samples)
        available_space = len(self.buffer) - self.write_idx

        if n > available_space:
            # Wrap around: fill the remaining space and write the rest at the start
            part1 = available_space
            self.buffer[self.write_idx :] = new_samples[:part1]
            part2 = n - part1
            self.buffer[:part2] = new_samples[part1:]
            self.write_idx = part2
        else:
            self.buffer[self.write_idx : self.write_idx + n] = new_samples
            self.write_idx += n

        if status:
            print(f"[AudioStreamer Warning] {status}")
        self.audio_buffer = (
            new_samples  # Store the latest block for visualization
        )

        if self.callback:
            audio_frame = indata.copy()
            if self.channels == 1:
                audio_frame = audio_frame.flatten()
            self.callback(audio_frame)

    def closeEvent(self, event):
        """Ensure clean shutdown of audio stream and widgets."""
        print("[MusicEditor] closeEvent triggered.")

        # 1. Stop the Audio Streamer
        if hasattr(self, "audio_streamer") and self.audio_streamer.running:
            print("[MusicEditor] Stopping AudioStreamer...")
            self.audio_streamer.stop()
            print("[MusicEditor] AudioStreamer stopped.")

        # 2. Explicitly close the WaveformWidget to trigger its closeEvent
        #    This helps ensure its worker thread is handled.
        if hasattr(self, "waveform_widget"):
            print("[MusicEditor] Closing WaveformWidget...")
            self.waveform_widget.close()
            print("[MusicEditor] WaveformWidget closed.")

        # 3. Call the base class closeEvent
        super().closeEvent(event)
        print("[MusicEditor] closeEvent finished.")

class ControlButton(QPushButton):
    def __init__(self, gold_icon_path, white_icon_path, parent=None):
        super().__init__(parent)
        self.gold_icon = QIcon(gold_icon_path)
        self.white_icon = QIcon(white_icon_path)
        self.setIcon(self.gold_icon)
        self.setIconSize(QSize(20, 20))
        self.setFlat(True)
        self.setStyleSheet("background-color: black; border: none;")
        self.hover_anim = QPropertyAnimation(self, b"geometry")
        self.hover_anim.setDuration(150)

    def enterEvent(self, event):
        rect = self.geometry()
        self.hover_anim.stop()
        self.hover_anim.setStartValue(rect)
        self.hover_anim.setEndValue(rect.adjusted(-2, -2, 2, 2))  # Enlarge slightly
        self.hover_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        rect = self.geometry()
        self.hover_anim.stop()
        self.hover_anim.setStartValue(rect)
        self.hover_anim.setEndValue(rect.adjusted(2, 2, -2, -2))
        self.hover_anim.start()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.setIcon(self.white_icon)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setIcon(self.gold_icon)
        super().mouseReleaseEvent(event)
