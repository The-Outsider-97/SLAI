# Updated synonym_trainer.py
import sys
import json
import torch
import math
import datetime
import numpy as np

from pathlib import Path
from typing import List, Tuple
from PyQt5.QtWidgets import QLabel, QFrame, QApplication , QHBoxLayout
from PyQt5.QtCore import QCoreApplication, pyqtSlot, QObject, QThread, pyqtSignal as QtCoreSignal, pyqtSignal
from PyQt5.QtSvg import QSvgWidget
from sklearn.metrics.pairwise import cosine_similarity

from models.slai_lm import SLAILM
from models.training.shared_signals import training_signals, safe_connect
from logs.logger import get_logger

logger = get_logger(__name__)

# === Configuration ===
STRUCTURED_WORDLIST_PATH = "src/agents/language/structured_wordlist_en.json"
EMBEDDING_PATH = "data/embeddings/glove.6B.200d.json"
FINAL_OUTPUT_PATH = Path("logs/enriched_wordlist_final.json")
PROGRESS_PATH = Path("logs/progress.json")
BATCH_SIZE = 10  # Ensure BATCH_SIZE is defined and > 0
MAX_SYNONYMS = 5
MAX_RELATED = 5

def load_embeddings(path: str) -> dict:
    """Loads embeddings from a JSON file."""
    embedding_path = Path(path)
    if not embedding_path.exists():
        # Try relative path from potential script location if absolute fails
        script_dir = Path(__file__).parent
        relative_path = script_dir.parent.parent / path # Adjust based on actual project structure
        if relative_path.exists():
             embedding_path = relative_path
        else:
            raise FileNotFoundError(f"Embedding file {path} not found!")
    try:
        with open(embedding_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {embedding_path}: {e}")
        raise ValueError(f"Invalid JSON format in {embedding_path}") from e
    except Exception as e:
        logger.error(f"Failed to load embeddings from {embedding_path}: {e}")
        raise

def load_progress() -> int:
    """Loads the last processed word index from the progress file."""
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as f:
                return json.load(f).get("last_word_index", 0)
        except json.JSONDecodeError:
            logger.warning(f"Progress file {PROGRESS_PATH} is corrupted. Starting from index 0.")
            return 0
        except Exception as e:
            logger.error(f"Error loading progress: {e}. Starting from index 0.")
            return 0
    return 0

def save_progress(word_index: int):
    """Saves the starting word index for the *next* batch."""
    progress_data = {
        "last_word_index": word_index, # Save the index of the next word to process
        "timestamp": datetime.datetime.now().isoformat(),
    }
    try:
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(PROGRESS_PATH, 'w') as f:
            json.dump(progress_data, f, indent=2)
        # Verification (optional but good practice)
        with open(PROGRESS_PATH, 'r') as f:
            saved_data = json.load(f)
            if saved_data.get("last_word_index") != word_index:
                 logger.error("Progress save verification failed! Data mismatch.")
                 raise IOError("Progress save verification failed") # Optional: raise error
    except Exception as e:
        logger.error(f"Progress save failed: {str(e)}")
        # Optionally re-raise or handle gracefully
        raise

def generate_from_embeddings(word: str, embedding_lookup: dict, topn=10) -> Tuple[List[str], List[str]]:
    """Generates synonyms and related terms using cosine similarity on embeddings."""
    if not isinstance(embedding_lookup, dict):
         logger.error("Embedding lookup is not a dictionary.")
         return [], []
    if word not in embedding_lookup:
        logger.warning(f"Word '{word}' not found in embeddings.")
        return [], []

    word_vec_data = embedding_lookup[word]
    if not isinstance(word_vec_data, list):
         logger.warning(f"Embedding for '{word}' is not a list: {type(word_vec_data)}")
         return [], []

    try:
        word_vec = np.array(word_vec_data).reshape(1, -1)
        embedding_dim = word_vec.shape[1]
    except Exception as e:
        logger.error(f"Error converting embedding for '{word}' to numpy array: {e}")
        return [], []

    valid_embeddings = {}
    for other, vec_data in embedding_lookup.items():
        if other == word: continue
        if isinstance(vec_data, list) and len(vec_data) == embedding_dim:
            try:
                valid_embeddings[other] = np.array(vec_data)
            except Exception:
                logger.warning(f"Skipping invalid embedding for '{other}'")
                pass # Skip words with invalid embedding format

    if not valid_embeddings:
        return [], []

    other_words = list(valid_embeddings.keys())
    # Ensure all vectors have the same dimension before stacking
    other_vecs_list = [vec for vec in valid_embeddings.values() if vec.shape == (embedding_dim,)]
    if not other_vecs_list:
        return [], []
    other_vecs = np.array(other_vecs_list)
    # Adjust other_words to match the filtered vectors
    other_words = [word for word, vec in valid_embeddings.items() if vec.shape == (embedding_dim,)]


    try:
        similarities = cosine_similarity(word_vec, other_vecs)[0]
        scores = {other_words[i]: similarities[i] for i in range(len(other_words))}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Filter synonyms and related terms based on thresholds
        synonyms = [w for w, score in ranked[:topn] if score > 0.65] # Adjusted threshold
        related = [w for w, score in ranked[len(synonyms):topn*2] if score > 0.5 and w not in synonyms] # Adjusted threshold and ensure no overlap

        # Limit the number returned
        synonyms = synonyms[:MAX_SYNONYMS]
        related = related[:MAX_RELATED]

        return synonyms, related
    except Exception as e:
        logger.error(f"Error in cosine similarity calculation for '{word}': {e}")
        return [], []

def save_enriched_wordlist(processed_data: dict):
    """Saves the fully processed (approved/modified) data to the final JSON file."""
    if not processed_data:
        logger.warning("No processed data to save.")
        return

    # Structure should match the original input format if possible
    # Assuming original format was like {"words": {word: entry}, "metadata": {...}, "version": ...}
    output_structure = {
        "words": processed_data,
        "metadata": {
            "source": "SLAI_SynonymTrainer_Output",
            "enrichment_method": "GloVe_Embeddings_User_Verified",
            "timestamp": datetime.datetime.now().isoformat(),
            "base_wordlist": STRUCTURED_WORDLIST_PATH # Record the source
        },
        "version": "1.1" # Example versioning
    }

    try:
        FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_structure, f, indent=2, ensure_ascii=False) # Use indent=2 for readability
        logger.info(f"Saved final enriched wordlist ({len(processed_data)} entries) to {FINAL_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save final enriched wordlist: {e}")
        training_signals.log_message.emit(f"[ERROR] Failed to save final output: {e}")


class EmbeddingManager:
    """Manages loading and querying word embeddings."""
    def __init__(self, path=EMBEDDING_PATH):
        try:
            self.embeddings = load_embeddings(path)
            logger.info(f"EmbeddingManager initialized with {len(self.embeddings)} embeddings.")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"EmbeddingManager initialization failed: {e}")
            self.embeddings = {} # Initialize empty to prevent crashes

    def get_similar(self, word, topn=10):
        """Retrieves similar words based on embeddings."""
        if not self.embeddings:
             logger.warning("Embeddings not loaded, cannot get similar words.")
             return [], []
        return generate_from_embeddings(word, self.embeddings, topn)

# If embedding generation or learning agent interaction is slow, move to a worker.
# For simplicity, let's assume for now it's fast enough, but this is where you'd add it.
class TrainingWorker(QObject):
    """
    Worker that processes training batches in a separate thread.
    Emits signals back to the main thread to safely update the UI.
    """
    # Signals to send results back to the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    update_batch = pyqtSignal(list, int, int)  # (display_list, current_batch_num, total_batches)
    error = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    @pyqtSlot()
    def process_and_display_current_batch(self):
        """
        Processes the current batch and emits results to the main thread.
        All UI updates must happen via emitted signals, not direct calls.
        """
        if not self.controller.is_running:
            self.error.emit("Controller is not running.")
            return

        if self.controller.current_batch_index == self.controller._last_processed_batch_index:
            # Prevent duplicate work
            return

        self.controller._last_processed_batch_index = self.controller.current_batch_index

        start_word_index = self.controller.current_batch_index * BATCH_SIZE
        end_word_index = min(start_word_index + BATCH_SIZE, len(self.controller.word_list_keys))
        current_batch_keys = self.controller.word_list_keys[start_word_index:end_word_index]

        self.controller.current_batch_entries = {}
        display_list = []

        current_batch_num = self.controller.current_batch_index + 1
        total_batches = self.controller.total_batches

        # Log start
        self.progress.emit(f"Processing Batch {current_batch_num}/{total_batches} (Words {start_word_index + 1}-{end_word_index})")

        try:
            for word in current_batch_keys:
                meta = self.controller.all_words_data.get(word, {})

                if not isinstance(meta, dict):
                    self.progress.emit(f"[SKIP] {word}: Invalid entry format.")
                    entry_for_display = {"pos": [], "synonyms": [], "related_terms": [], "status": "skipped"}
                    self.controller.current_batch_entries[word] = entry_for_display.copy()
                    display_list.append((word, entry_for_display))
                    continue

                meta.setdefault('pos', [])
                meta.setdefault('synonyms', [])
                meta.setdefault('related_terms', [])

                new_synonyms, new_related = self.controller.embedding_manager.get_similar(word)

                entry_for_display = {
                    "pos": meta["pos"],
                    "synonyms": new_synonyms,
                    "related_terms": new_related,
                    "original_synonyms": meta["synonyms"].copy(),
                    "original_related": meta["related_terms"].copy(),
                    "status": "pending"
                }
                self.controller.current_batch_entries[word] = entry_for_display.copy()
                display_list.append((word, entry_for_display))

                self.progress.emit(f"Prepared word: {word}")

            # Send the whole batch to the main thread for UI update
            self.update_batch.emit(display_list, current_batch_num, total_batches)

        except Exception as e:
            self.error.emit(f"Error in batch processing: {str(e)}")

        # Signal completion
        self.finished.emit()

# === Training Controller Class ===
class SynonymTrainerController(QObject):
    # Assuming LanguageEditor is correctly imported
    # from models.training.language_editor import LanguageEditor # Uncomment if needed

    def __init__(self, language_editor, parent=None): # Removed type hint temporarily if causing import issues
        # Assuming LearningAgent is correctly imported or mocked
        from src.agents.learning_agent import LearningAgent # Uncomment if needed
        super().__init__(parent)
        self.language_editor = language_editor
        self.all_words_data = {} # Store original data {word: entry}
        self.word_list_keys = [] # Ordered list of words (keys)
        self.embedding_manager = EmbeddingManager() # Use the manager
        self.current_batch_index = 0 # Index of the batch
        self.current_batch_entries = {} # Stores {word: modified_entry} for the batch being reviewed
        self.processed_data = {} # Accumulates approved data {word: final_entry}
        self.total_batches = 0
        self.is_running = False
        self._last_processed_batch_index = -1 # For duplicate log prevention
        self.worker = TrainingWorker(self)

        # --- Mock LearningAgent if not available ---
        try:
            from src.agents.learning_agent import LearningAgent
            self.learning_agent = LearningAgent(
                shared_memory=None, agent_factory=None, slai_lm=None,
                safety_agent=None, env=None, config={}
            )
            self.update_counter = 0
        except ImportError:
            logger.warning("LearningAgent not found, embedding feedback will not be processed.")
            self.learning_agent = None
            self.update_counter = -1 # Disable feedback processing

        self._connect_signals()

    def _connect_signals(self):
        """Connects signals between the controller and the UI editor."""
        if not self.language_editor:
            logger.error("LanguageEditor instance not provided to SynonymTrainerController.")
            return

        safe_connect(
            self.worker.update_batch,
            lambda display_list, current_batch, total_batches: (
                self.language_editor.display_batch(display_list),
                self.language_editor.update_batch_counter(current_batch, total_batches),
                self.language_editor.update_progress(int((current_batch / total_batches) * 100) if total_batches > 0 else 0)
            )
        )

        # Connect signals from LanguageEditor UI to controller slots
        safe_connect(self.language_editor.nextBatchClicked, self.next_batch)
        safe_connect(self.language_editor.prevBatchClicked, self.previous_batch)
        safe_connect(self.language_editor.approveBatchClicked, self.approve_batch)
        safe_connect(self.language_editor.rejectBatchClicked, self.reject_batch)
        safe_connect(self.language_editor.shuffleBatchClicked, self.shuffle_batch)
        safe_connect(self.language_editor.editor_signals.term_decision, self.handle_term_update)

        # --- Connections FROM Controller/Shared Signals TO UI (LanguageEditor) ---
        # Log messages are a prime candidate for safe_connect if emitted rapidly
        safe_connect(training_signals.log_message, self.language_editor.append_log)
        # Batch status updates the UI, use safe_connect
        safe_connect(training_signals.batch_status, self.language_editor.update_batch_status)
        if self.learning_agent:
            safe_connect(training_signals.embedding_update, self.handle_embedding_update)

    def load_data(self):
        """Loads wordlist, embeddings, and progress state."""
        try:
            # Load structured wordlist
            wordlist_path = Path(STRUCTURED_WORDLIST_PATH)
            if not wordlist_path.exists():
                 raise FileNotFoundError(f"Wordlist file not found: {STRUCTURED_WORDLIST_PATH}")
            with open(wordlist_path, "r", encoding="utf-8") as f:
                swl_raw = json.load(f)

            if "words" not in swl_raw or not isinstance(swl_raw["words"], dict):
                 training_signals.log_message.emit("[ERROR] Invalid structured wordlist format.")
                 logger.error("Invalid structured wordlist format: 'words' key missing or not a dict.")
                 return False

            self.all_words_data = swl_raw["words"]
            self.word_list_keys = list(self.all_words_data.keys()) # Get ordered list of words

            # Load embeddings (handled by EmbeddingManager constructor)
            if not self.embedding_manager.embeddings:
                 training_signals.log_message.emit("[ERROR] Failed to load embeddings.")
                 return False

            # Load progress (word index) and calculate batch index
            last_word_index = load_progress()
            if BATCH_SIZE <= 0:
                 logger.error("BATCH_SIZE must be greater than 0.")
                 return False
            # Correctly calculate the starting batch index
            self.current_batch_index = last_word_index // BATCH_SIZE

            self.total_batches = math.ceil(len(self.word_list_keys) / BATCH_SIZE)

            # Load previously processed data if FINAL_OUTPUT_PATH exists
            if FINAL_OUTPUT_PATH.exists():
                try:
                    with open(FINAL_OUTPUT_PATH, "r", encoding="utf-8") as f:
                        loaded_final_data = json.load(f)
                        # Ensure it's the expected structure
                        if isinstance(loaded_final_data, dict) and "words" in loaded_final_data:
                             self.processed_data = loaded_final_data["words"]
                             logger.info(f"Loaded {len(self.processed_data)} previously processed entries.")
                        else:
                             logger.warning(f"Final output file {FINAL_OUTPUT_PATH} has unexpected format. Starting fresh.")
                             self.processed_data = {}
                except json.JSONDecodeError:
                    logger.warning(f"Final output file {FINAL_OUTPUT_PATH} is corrupted. Starting fresh.")
                    self.processed_data = {}
                except Exception as e:
                    logger.error(f"Error loading existing final output: {e}. Starting fresh.")
                    self.processed_data = {}
            else:
                 self.processed_data = {}


            training_signals.log_message.emit(
                f"[INIT] Loaded {len(self.word_list_keys)} words. Total batches: {self.total_batches}. "
                f"Resuming at batch {self.current_batch_index + 1} (Word index {last_word_index}). "
                f"Loaded {len(self.processed_data)} existing approved entries."
            )
            return True
        except FileNotFoundError as e:
            training_signals.log_message.emit(f"[ERROR] Failed to load data file: {e}")
            logger.error(f"Failed to load data file: {e}")
            return False
        except Exception as e:
            training_signals.log_message.emit(f"[ERROR] An unexpected error occurred during loading: {e}")
            logger.exception("An unexpected error occurred during loading.") # Log traceback
            return False

    def start_training(self):
        """Starts the training loop."""
        if not self.load_data():
            self.is_running = False
            logger.error("Training cannot start due to data loading errors.")
            training_signals.log_message.emit("[ERROR] Training cannot start. Check logs.")
            return
        self.is_running = True
        self._last_processed_batch_index = -1 # Reset log duplicate prevention
        self.worker.process_and_display_current_batch()
        training_signals.log_message.emit("[SYSTEM] Controller started. Displaying first batch.")

    def stop_training(self):
        """Stops the training loop and saves progress."""
        if not self.is_running:
            return
        self.is_running = False
        current_word_index = self.current_batch_index * BATCH_SIZE
        save_progress(current_word_index) # Save the starting index of the current batch
        training_signals.log_message.emit(f"[STOPPED] Training stopped. Progress saved for word index {current_word_index}.")
        logger.info(f"Training stopped. Progress saved for word index {current_word_index}.")
        # Optionally save the currently accumulated processed_data as well
        # save_enriched_wordlist(self.processed_data)



    @pyqtSlot(str, str, str, str)
    def handle_term_update(self, word: str, term_type: str, original_term: str, decision: str):
        """Handles decisions (keep/reject/modify) for individual terms from the UI."""
        if not self.is_running: return
        if word not in self.current_batch_entries:
            logger.warning(f"Received term update for word '{word}' not in current batch.")
            return

        entry = self.current_batch_entries[word]
        term_list_key = "synonyms" if term_type == "synonym" else "related_terms"

        # --- Update the entry in current_batch_entries based on decision ---
        current_terms = entry.get(term_list_key, [])
        modified = False

        if decision == 'reject':
            if original_term in current_terms:
                current_terms.remove(original_term)
                modified = True
        elif decision == 'keep':
            # 'keep' usually means no change needed unless the term was somehow removed previously
            if original_term not in current_terms:
                 current_terms.append(original_term) # Add back if missing
                 modified = True
        else: # Modification or Addition
            is_addition = not original_term # True if original_term is empty, indicating an added term
            new_term = decision # The decision string is the new term

            if is_addition:
                if new_term not in current_terms:
                    current_terms.append(new_term)
                    modified = True
            else: # Modification of an existing term
                if original_term in current_terms:
                    idx = current_terms.index(original_term)
                    if current_terms[idx] != new_term:
                        current_terms[idx] = new_term
                        modified = True
                elif new_term not in current_terms: # Original term wasn't there, but add the new one
                    current_terms.append(new_term)
                    modified = True

        if modified:
            entry[term_list_key] = current_terms # Update the list in the entry
            entry["status"] = "modified" # Mark entry as modified
            logger.debug(f"Updated entry for '{word}': {entry}")


        # --- Learning Agent Feedback ---
        if self.learning_agent and self.update_counter != -1:
             # Prepare feedback signal data
             feedback_data = {
                 'word': word,
                 'term_type': term_type,
                 'original_term': original_term,
                 'decision': decision # Pass the user's action
             }
             # Emit the signal for the learning agent (or potentially other listeners)
             training_signals.embedding_update.emit(feedback_data)

    @pyqtSlot()
    def approve_batch(self):
        """User approves the entire batch with modifications made."""
        if not self.is_running: return
        training_signals.log_message.emit(f"Approving Batch {self.current_batch_index + 1}...")
        training_signals.batch_status.emit("APPROVED") # Signal to UI

        # Add all entries from the current reviewed batch to the final processed data
        for word, entry in self.current_batch_entries.items():
             if entry["status"] != "skipped":
                 # Prepare final entry: remove temporary keys like 'status', 'original_*'
                 final_entry = {
                     "pos": entry.get("pos", []), # Use .get for safety
                     "synonyms": entry.get("synonyms", []),
                     "related_terms": entry.get("related_terms", [])
                 }
                 self.processed_data[word] = final_entry # Add/overwrite in final dataset

        # Optionally save the entire processed data periodically or just on finalize
        save_enriched_wordlist(self.processed_data) # Maybe save less frequently?

        # Check if this is the last batch
        if (self.current_batch_index + 1) >= self.total_batches:
             self.finalize_and_save()
        else:
            # Proceed to the next batch
            self.current_batch_index += 1
            next_word_index = self.current_batch_index * BATCH_SIZE
            save_progress(next_word_index) # Save progress for the start of the next batch
            self._last_processed_batch_index = -1 # Allow next batch logs
            self.worker.process_and_display_current_batch()

    @pyqtSlot()
    def reject_batch(self):
        """User rejects the entire batch. Skip and move to next."""
        if not self.is_running: return
        training_signals.log_message.emit(f"[REJECTED] Skipping Batch {self.current_batch_index + 1}")
        training_signals.batch_status.emit("REJECTED") # Signal to UI
        # Don't add current_batch_entries to processed_data

        if (self.current_batch_index + 1) < self.total_batches:
            self.current_batch_index += 1
            next_word_index = self.current_batch_index * BATCH_SIZE
            save_progress(next_word_index) # Save progress for the start of the next batch
            self._last_processed_batch_index = -1 # Allow next batch logs
            self.worker.process_and_display_current_batch()
        else:
            # Last batch rejected, finalize with existing processed data
            training_signals.log_message.emit("Last batch rejected. Finalizing with previously approved data.")
            self.finalize_and_save()

    @pyqtSlot()
    def next_batch(self):
        """Move to the next batch without approving the current one (acts like reject)."""
        if not self.is_running: return
        if (self.current_batch_index + 1) < self.total_batches:
             training_signals.log_message.emit(f"Moving to next batch (Current Batch {self.current_batch_index + 1} NOT approved).")
             self.current_batch_index += 1
             next_word_index = self.current_batch_index * BATCH_SIZE
             save_progress(next_word_index) # Save progress for the start of the next batch
             self._last_processed_batch_index = -1 # Allow next batch logs
             self.worker.process_and_display_current_batch()
        else:
             training_signals.log_message.emit("Already at the last batch.")
             # Optionally finalize if user confirms, or just indicate end
             self.finalize_and_save() # Or add a confirmation dialog

    @pyqtSlot()
    def previous_batch(self):
        """Move to the previous batch."""
        if not self.is_running: return
        if self.current_batch_index > 0:
             # Changes to the current batch are lost
             training_signals.log_message.emit(f"Moving to previous batch (Changes to Batch {self.current_batch_index + 1} discarded).")
             self.current_batch_index -= 1
             next_word_index = self.current_batch_index * BATCH_SIZE # Index for start of the batch we are going TO
             save_progress(next_word_index) # Save progress
             self._last_processed_batch_index = -1 # Allow logs for prev batch
             # Re-process and display the previous batch
             # Note: This does not restore user modifications from that previous batch if they navigated away.
             # It just shows the generated terms for that batch again.
             self.worker.process_and_display_current_batch()
        else:
             training_signals.log_message.emit("Already at the first batch.")

    @pyqtSlot()
    def shuffle_batch(self):
         """Shuffles the order of word keys and reloads the current batch index."""
         if not self.is_running: return
         training_signals.log_message.emit("Shuffling word order...")
         np.random.shuffle(self.word_list_keys) # Shuffle the list of keys in place
         # Keep the current batch index, but reload the content based on the shuffled list
         self._last_processed_batch_index = -1 # Allow logs
         self.worker.process_and_display_current_batch() # Display the new content for the current index
         training_signals.log_message.emit("Word order shuffled. Displaying current batch with new order.")


    def finalize_and_save(self):
        """Saves the accumulated processed_data to the final output file."""
        self.is_running = False
        training_signals.log_message.emit(f"Finalizing... Saving {len(self.processed_data)} approved/modified entries.")
        logger.info(f"Finalizing training. Saving {len(self.processed_data)} entries.")

        save_enriched_wordlist(self.processed_data) # Call the save function

        # Reset progress only if desired after successful completion
        save_progress(0)
        logger.info("Training complete. Progress reset.")
        training_signals.log_message.emit("[SUCCESS] Training complete. Final data saved.")

    def cleanup(self):
        """Cleans up data structures (e.g., if restarting)."""
        self.processed_data.clear()
        self.current_batch_entries.clear()
        self.all_words_data = {}
        self.word_list_keys = []
        self.current_batch_index = 0
        self.total_batches = 0
        self.is_running = False
        self._last_processed_batch_index = -1
        logger.info("SynonymTrainerController cleaned up.")


    def handle_embedding_update(self, feedback_data: dict):
        """Processes feedback data to potentially train the LearningAgent."""
        if not self.learning_agent or self.update_counter == -1:
             return # Learning agent not available or disabled

        word = feedback_data.get('word')
        decision = feedback_data.get('decision')
        # original_term = feedback_data.get('original_term') # Could be used for more complex feedback

        if not word or not decision:
             logger.warning("Incomplete feedback data received for embedding update.")
             return

        if word in self.embedding_manager.embeddings:
            try:
                 embedding = np.array(self.embedding_manager.embeddings[word])
                 # Simple feedback: 1 for keep/modified, 0 for reject
                 label = 1 if decision != 'reject' else 0

                 # Pass embedding and label to learning agent
                 embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                 label_tensor = torch.tensor([label], dtype=torch.float32)
                 
                 combined_tensor = torch.cat([embedding_tensor, label_tensor])
                 
                 self.learning_agent.observe(combined_tensor)
                 self.update_counter += 1

                 # Trigger training periodically
                 if self.update_counter >= 20: # Train every 20 feedback points
                      self.learning_agent.train_from_embeddings() # Assuming this method exists
                      self.update_counter = 0
                      training_signals.log_message.emit("[Learning] LearningAgent updated with user feedback.")
                      logger.info("[Learning] LearningAgent model updated based on user feedback.")

            except Exception as e:
                 logger.error(f"Error processing embedding feedback for '{word}': {e}")

def run_synonym_training():
    from models.training.language_editor import LanguageEditor
    shared_memory = {}
    slai_lm = SLAILM(shared_memory)

    swl_raw = slai_lm.structured_wordlist
    if not isinstance(swl_raw, dict):
        training_signals.log_message.emit("[ERROR] structured_wordlist is not a dict")
        return

    if "words" not in swl_raw or not isinstance(swl_raw["words"], dict):
        training_signals.log_message.emit("[ERROR] structured_wordlist 'words' key is missing or invalid")
        return

    words = list(swl_raw["words"].items())
    embedding_data = load_embeddings(EMBEDDING_PATH)
    start_index = load_progress()
    total_words = len(words)

    preview_dict = {"words": {k: v for k, v in words[:5]}}
    preview_json = json.dumps(preview_dict, ensure_ascii=False, indent=2)
    training_signals.log_message.emit(
        f"[INIT] Loaded {total_words} words from structured wordlist (resuming at index {start_index}).\n"
        f"Preview of loaded words:\n{preview_json}"
    )

    # Signal-based synchronization flags
    current_batch_approved = False
    should_continue = True

    def handle_batch_approval(approved):
        nonlocal current_batch_approved
        current_batch_approved = approved

    def handle_continuation(continue_flag):
        nonlocal should_continue
        should_continue = continue_flag

    # Connect signals
    training_signals.batch_approved.connect(handle_batch_approval)
    training_signals.batch_approved.connect(handle_continuation)

    for batch_start in range(start_index, total_words, BATCH_SIZE):
        if not should_continue:
            break

        batch = words[batch_start:batch_start+BATCH_SIZE]
        current_batch_approved = True  # Assume approval until rejected

        # Process batch
        for word, meta in batch:
            if not isinstance(meta, dict):
                training_signals.log_message.emit(f"[SKIP] {word}: entry is not a dict: {meta}")
                continue

            corrected = False
            for key in ['pos', 'synonyms', 'related_terms']:
                if key not in meta:
                    meta[key] = []
                    corrected = True
                elif isinstance(meta[key], str):
                    meta[key] = [meta[key]]
                    corrected = True
                elif not isinstance(meta[key], list):
                    meta[key] = []
                    corrected = True

            if corrected:
                training_signals.log_message.emit(f"[AUTO-CORRECT] {word}: corrected entry structure.")

            if len(meta["synonyms"]) >= MAX_SYNONYMS:
                training_signals.log_message.emit(f"[SKIP] {word}: already has sufficient synonyms")
                continue

            synonyms, related = generate_from_embeddings(word, embedding_data)
            entry = {
                "pos": meta["pos"],
                "synonyms": synonyms,
                "related_terms": related
            }

            if isinstance(entry["pos"], list):
                # Emit approval request and wait via event loop
                training_signals.approval_request.emit(word, entry, "synonym_block")

            # Process events while waiting
            while not current_batch_approved:
                QCoreApplication.processEvents()

            if not current_batch_approved:
                training_signals.log_message.emit(f"[REJECTED] Batch starting at {batch_start}")
                break

            # Save approved entry immediately
            append_log(word, entry)

            # MODIFIED APPROVAL HANDLING
            approval_data = {
                'word': word,
                'entry': entry,
                'batch_position': batch_start,
                'total_batches': math.ceil(total_words/BATCH_SIZE)
            }
            
            # Emit signal for GUI approval
            training_signals.approval_request.emit(approval_data)

    app = QApplication(sys.argv)
    editor = LanguageEditor()
    controller = SynonymTrainerController(editor)

    editor.show()
    controller.start_training()

    sys.exit(app.exec_())

    # Prompt for continuation
    training_signals.approval_request.emit(
        "Continue?", 
        {"remaining": total_words - (batch_start + BATCH_SIZE)},
        "batch_continuation"
    )

    # Request continuation
    training_signals.batch_continuation.emit(
        new_progress, 
        total_words - new_progress
    )
    
    # Wait for continuation decision
    should_continue = True  # Reset flag
    while should_continue is None:
        QCoreApplication.processEvents()

    # Cleanup connections
    training_signals.batch_approved.disconnect(handle_batch_approval)
    training_signals.batch_approved.disconnect(handle_continuation)
    training_signals.log_message.emit("[TRAINING] Synonym training completed")

def append_log(word: str, entry: dict):
    """Read-only mock function"""
    logger.info(f"Read-only mode: Would save {word} entry")
