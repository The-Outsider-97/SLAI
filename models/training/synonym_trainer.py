import json
import time
import os
import math
import datetime
import numpy as np
import logging as logger

from pathlib import Path
from typing import List, Tuple
from PyQt5.QtCore import QCoreApplication
from sklearn.metrics.pairwise import cosine_similarity

from models.slai_lm import SLAILM
from models.training.shared_signals import training_signals

# === Configuration ===
STRUCTURED_WORDLIST_PATH = "src/agents/language/structured_wordlist_en.json"
EMBEDDING_PATH = "data/embeddings/glove.6B.100d.json"
LOG_OUTPUT_PATH = Path("logs/enriched_wordlist.txt")
PROGRESS_PATH = Path("logs/progress.json")
# SAVE_INTERVAL = 60
BATCH_SIZE = 10
MAX_SYNONYMS = 5
MAX_RELATED = 5

def load_embeddings(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Embedding file {path} not found!")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_progress() -> int:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_progress(index: int):
    progress_data = {
        "last_index": index,
        "timestamp": datetime.now().isoformat(),
        "total_processed": index
    }
    
    try:
        with open(PROGRESS_PATH, 'w') as f:
            json.dump(progress_data, f)
        # Verify write
        with open(PROGRESS_PATH, 'r') as f:
            saved = json.load(f)
            if saved['last_index'] != index:
                raise IOError("Progress save verification failed")
    except Exception as e:
        logger.error(f"Progress save failed: {str(e)}")
        raise

def append_log(word: str, entry: dict):
    LOG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({word: entry}, ensure_ascii=False) + "\n")

def generate_from_embeddings(word: str, embedding_lookup: dict, topn=10) -> Tuple[List[str], List[str]]:
    if word not in embedding_lookup:
        return [], []
    word_vec = np.array(embedding_lookup[word]).reshape(1, -1)
    scores = {
        other: cosine_similarity(word_vec, np.array(vec).reshape(1, -1))[0][0]
        for other, vec in embedding_lookup.items() if other != word
    }
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    synonyms = [w for w, _ in ranked[:MAX_SYNONYMS]]
    related = [w for w, _ in ranked[MAX_SYNONYMS:MAX_SYNONYMS + MAX_RELATED]]

    retries = 3
    for attempt in range(retries):
        try:
            synonyms, related = generate_from_embeddings(word, training_signals.embedding_data)
            break
        except Exception as e:
            training_signals.log_message.emit(f"[RETRY {attempt+1}] Failed generating for '{word}': {e}")
            synonyms, related = [], []
    return synonyms, related

def run_synonym_training():
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
            #while not current_batch_approved:
            #    QCoreApplication.processEvents()

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

        # After batch completion
        new_progress = batch_start + BATCH_SIZE
        progress_percent = int((new_progress / total_words) * 100)
        
        # Update progress
        save_progress(new_progress)
        training_signals.log_message.emit(f"\nCompleted batch {batch_start//BATCH_SIZE + 1}/"
                                        f"{math.ceil(total_words/BATCH_SIZE)}")

        # Request continuation
        training_signals.batch_continuation.emit(
            new_progress, 
            total_words - new_progress
        )
        
        # Wait for continuation decision
        should_continue = True  # Reset flag
        #while should_continue is None:
        #    QCoreApplication.processEvents()

    # Cleanup connections
    training_signals.batch_approved.disconnect(handle_batch_approval)
    training_signals.batch_approved.disconnect(handle_continuation)
    training_signals.log_message.emit("[TRAINING] Synonym training completed")

def _handle_batch_completion(self, batch_start, batch_size, total_words):
    # Update progress
    save_progress(min(batch_start + batch_size, total_words))
    
    # Update UI
    training_signals.log_message.emit(
        f"\nCompleted batch {batch_start//BATCH_SIZE + 1}/"
        f"{math.ceil(total_words/BATCH_SIZE)}"
    )
    
    # Prompt for continuation
    training_signals.approval_request.emit(
        "Continue?", 
        {"remaining": total_words - (batch_start + BATCH_SIZE)},
        "batch_continuation"
    )

def append_log(word: str, entry: dict):
    """Read-only mock function"""
    logger.info(f"Read-only mode: Would save {word} entry")
