import json
import numpy as np
import torch
import hashlib
import logging
import threading

from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QObject, QCoreApplication
#from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication

# Local imports
from src.utils.data_loader import DataLoader
from src.agents.perception_agent import PerceptionAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.learning_agent import NaNException, GradientExplosionError
from src.collaborative.shared_memory import SharedMemory
from models.slai_lm import SLAILMValueModel
from data.multimodal_dataset import MultimodalDataset
from logs.logger import get_logger

logger = get_logger(__name__)

# Metrics
from evaluate import load as load_metric
bertscore = load_metric("bertscore")
bleu = load_metric("bleu")
rouge = load_metric("rouge")

# === Constants ===
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
HISTORY_SIZE = 100

# === Helper Functions ===
def save_checkpoint(agent, phase, epoch, metrics):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = CHECKPOINT_DIR / f"{phase}_epoch{epoch}_{timestamp}.pt"
    torch.save({
        "agent_state": agent.params,
        "optimizer": agent.optimizer,
        "metrics": metrics
    }, ckpt_path)
    return ckpt_path

def load_checkpoint(agent, ckpt_path):
    ckpt = torch.load(ckpt_path)
    agent.load_params(ckpt["agent_state"])
    return ckpt["metrics"]

def simple_bleu(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    matches = sum(1 for token in cand_tokens if token in ref_tokens)
    return matches / max(len(cand_tokens), 1)

def simple_rouge(reference: str, candidate: str) -> float:
    ref_ngrams = set(zip(reference.split(), reference.split()[1:]))
    cand_ngrams = set(zip(candidate.split(), candidate.split()[1:]))
    overlap = ref_ngrams & cand_ngrams
    return len(overlap) / max(len(ref_ngrams), 1)

seen_hashes = set()  # used to avoid duplicate entries


def compute_entry_hash(word, entry):
    """Compute a stable hash for a word-entry block to detect duplicates."""
    key_data = json.dumps({"word": word, **entry}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()

def human_approve(samples: list, approval_handler, max_rejects=3) -> bool:
    rejected = 0

    for sample in samples:
        word = sample.get("word")
        entry = sample.get("entry", {})
        entry_hash = compute_entry_hash(word, entry)
        if entry_hash in seen_hashes:
            logger.info(f"[Skipped: {word}] Already processed.")
            continue

        seen_hashes.add(entry_hash)

        lines = [f"{word}:", f"  pos: {entry.get('pos', [])}", "  synonyms:"]
        lines += [f"    - {s}" for s in entry.get("synonyms", [])]
        lines += ["  related_terms:"]
        lines += [f"    - {r}" for r in entry.get("related_terms", [])]

        pretty_output = "\n".join(lines)
        logger.info(f"[TRAINING] {pretty_output}")

        corrected_synonyms = []
        corrected_related = []

        for synonym in entry.get("synonyms", []):
            approval_handler.request_approval.emit(word, synonym, "synonym")
            while synonym not in approval_handler.gui_decisions:
                QApplication.processEvents()
            result = approval_handler.gui_decisions.pop(synonym)
            if result == 'y':
                corrected_synonyms.append(synonym)
            elif result == 'n':
                rejected += 1
            elif result:
                corrected_synonyms.append(result)

        for term in entry.get("related_terms", []):
            approval_handler.request_approval.emit(word, term, "related_term")
            while term not in approval_handler.gui_decisions:
                QApplication.processEvents()
            result = approval_handler.gui_decisions.pop(term)
            if result == 'y':
                corrected_related.append(term)
            elif result == 'n':
                rejected += 1
            elif result:
                corrected_related.append(result)

        entry["synonyms"] = corrected_synonyms
        entry["related_terms"] = corrected_related

        logger.info("\u2713 Entry saved and logged.\n" + "-" * 40)

    return rejected <= max_rejects


# === Core Training ===
class SLAITrainer:
    def __init__(self, agent, shared_memory, target, response, agent_factory, approval_handler, dist_config=None):
        self.dist_config = dist_config or {}
        if self.dist_config.get('enabled', False):
            self._setup_distributed()

        self.agent = agent
        self.shared_memory = shared_memory
        self.approval_handler = approval_handler
        self.knowledge = KnowledgeAgent(shared_memory, agent_factory)
        self.value_model = SLAILMValueModel(agent.slai_lm)
        self.history = deque(maxlen=HISTORY_SIZE)
        bleu_score = simple_bleu(target, response)
        rouge_score = simple_rouge(target, response)
        
        # Training state
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.grad_clip = 1.0

    def run(self):
        from models.training.synonym_trainer import run_synonym_training
        run_synonym_training()

    def _setup_distributed(self):
        """Initialize distributed training environment"""
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=self.dist_config['init_method'],
            world_size=self.dist_config['world_size'],
            rank=self.dist_config['rank']
        )
        self.agent.model = torch.nn.parallel.DistributedDataParallel(
            self.agent.model,
            device_ids=[self.dist_config['local_rank']]
        )
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            num_replicas=self.dist_config['world_size'],
            rank=self.dist_config['rank']
        )

        dist_config = {
            'enabled': True,
            'init_method': 'tcp://localhost:23456',
            'world_size': 4,
            'rank': 0,
            'local_rank': 0,
            'batch_size': 32
        }
        
        trainer = SLAITrainer(agent, shared_memory, dist_config=dist_config)

# === Main Trainer ===
    def pretrain(self, dataset, epochs=10, lr=1e-3):
        
        """Multimodal pretraining with reconstruction losses and distributed training"""
        if self.dist_config.get('enabled', False):
            dataset = DataLoader(
                dataset,
                sampler=self.sampler,
                batch_size=self.dist_config['batch_size']
            )

        self.agent.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(dataset, desc=f"Pretrain Epoch {epoch+1}"):
                # [1] Encode batch through perception agent
                audio_input = {"audio": batch["audio"]}
                embedding = self.agent.perception_agent.forward(audio_input)
                self.shared_memory.set("audio_embedding", embedding)
                
                # [2] Pass embedding to learning agent
                embedding = self.shared_memory.get("audio_embedding")
                if embedding is not None:
                    self.agent.observe(embedding)

                masked_batch = self.agent.pretrainer.masked_modality_modeling(batch) # Multimodal masking
                losses_dict = self.agent.pretrainer.masked_modality_modeling(masked_batch) # Forward pass
                
                # Total loss
                loss = sum(losses_dict.values())
                if not np.isfinite(loss.item()):
                    raise NaNException("Detected NaN during loss calculation.")
                
                if isinstance(loss, torch.Tensor):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.params, self.grad_clip)
                    if grad_norm > 1000:
                        raise GradientExplosionError(norm=grad_norm)

                # Backprop
                self.agent.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.params, self.grad_clip)
                self.agent.step(lr)
                
                epoch_loss += loss.item()
                self.shared_memory.append("pretrain_loss", loss.item())

            # Training step
            if self.dist_config.get('enabled', False):
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss /= self.dist_config['world_size']
            
            # Save checkpoints only on main process    
            if not self.dist_config.get('enabled', False) or self.dist_config['rank'] == 0:
                avg_loss = epoch_loss / len(dataset)
                losses.append(avg_loss)
                save_checkpoint(self.agent, "pretrain", epoch, {"loss": avg_loss})
            
            # Human validation
            if epoch % 5 == 0:
                samples = [{"prompt": "Explain gravity", "response": self.agent.generate_response("Explain gravity")}]
                if not human_approve(samples, self.approval_handler):
                    print("Pretraining halted by human oversight")
                    break

    
        return {"phase": "pretrain", "losses": losses}

# === Phase II — Finetune ===
    def finetune(self, dataset, epochs=5, lr=5e-4):
        """Task-specific fine-tuning with quality metrics"""

        self.agent.train()
        metrics = {"loss": [], "bleu": [], "rouge": []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            for prompt, target in tqdm(dataset, desc=f"Finetune Epoch {epoch+1}"):
                # Augment with knowledge
                context = self.knowledge.retrieve(prompt)[:2]
                augmented_prompt = f"Context: {context}\n\nQuestion: {prompt}"
                
                # Generate response
                response = self.agent.generate_response(augmented_prompt)
                
                # Calculate metrics
                with torch.no_grad():
                    bleu_score = bleu.compute(predictions=[response], references=[target])["bleu"]
                    rouge_score = rouge.compute(predictions=[response], references=[target])["rougeL"]
                    
                # Backprop from BLEU
                loss = 1 - bleu_score
                if not np.isfinite(loss.item()):
                    raise NaNException("Detected NaN during loss calculation.")
                
                if isinstance(loss, torch.Tensor):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.params, self.grad_clip)
                    if grad_norm > 1000:
                        raise GradientExplosionError(norm=grad_norm)

                self.agent.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.params, self.grad_clip)
                self.agent.step(lr)
                
                # Track metrics
                epoch_loss += loss.item()
                metrics["bleu"].append(bleu_score)
                metrics["rouge"].append(rouge_score)
                self.shared_memory.append("finetune_loss", loss.item())
            
            # Epoch metrics
            avg_loss = epoch_loss / len(dataset)
            metrics["loss"].append(avg_loss)
            save_checkpoint(self.agent, "finetune", epoch, metrics)
            
            # Human checkpoint
            samples = [{
                "prompt": prompt,
                "response": self.agent.generate_response(prompt)
            } for prompt, _ in dataset[:3]]
            
            if not human_approve(samples, self.approval_handler):
                print("Finetuning halted by human oversight")
                break
        
        return {"phase": "finetune", "metrics": metrics}

    def get_music_training_data(self):
        """Convert user edits to training pairs"""
        raw_data = self.shared_memory.get("training_data", [])
        return [(item['original'], item['modified']) for item in raw_data]

    def finetune_music_model(self):
        dataset = self.get_music_training_data()
        if len(dataset) >= 100:  # Minimum batch size
            return self.finetune(dataset, epochs=3, lr=1e-4)

# === Phase III — RLHF ===
    def rlhf(self, dataset, epochs=3, lr=1e-5):
        """Human-aligned RL with reward modeling"""

        self.agent.train()
        rewards = []
        
        for epoch in range(epochs):
            epoch_reward = 0
            for prompt in tqdm(dataset, desc=f"RLHF Epoch {epoch+1}"):
                # Generate candidate responses
                candidates = [self.agent.generate_response(prompt) for _ in range(4)]
                
                # Get rewards
                with torch.no_grad():
                    reward_scores = self.value_model.score_trajectory({
                        "input": [prompt]*4,
                        "output": candidates
                    })
                
                # Update projection layer
                self.agent.update_projection(reward_scores, lr)
                
                # Track metrics
                epoch_reward += np.mean(reward_scores)
                rewards.append(reward_scores)
                self.shared_memory.append("rlhf_rewards", reward_scores)

            # Save checkpoint
            save_checkpoint(self.agent, "rlhf", epoch, {"reward": epoch_reward/len(dataset)})
            
            # Human evaluation
            if epoch % 2 == 0:
                samples = [{
                    "prompt": prompt,
                    "response": self.agent.generate_response(prompt)
                } for prompt in dataset[:3]]
                
                if not human_approve(samples, self.approval_handler):
                    print("RLHF training halted by human oversight")
                    break
        if not np.isfinite(loss.item()):
            raise NaNException("Detected NaN during loss calculation.")
        
        if isinstance(loss, torch.Tensor):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.params, self.grad_clip)
            if grad_norm > 1000:
                raise GradientExplosionError(norm=grad_norm)
                    
        return {"phase": "rlhf", "rewards": rewards}

# === Main Execution ===
def train_slaim(phase_configs, agent, dataset, shared_memory):
    trainer = SLAITrainer(agent, shared_memory)
    results = {}
    
    for phase, config in phase_configs.items():
        print(f"\n=== Starting {phase.upper()} Phase ===")
        match phase:
            case "pretrain":
                result = trainer.pretrain(dataset, **config)
            case "finetune":
                result = trainer.finetune(dataset, **config)
            case "rlhf":
                result = trainer.rlhf(dataset, **config)
            case _:
                raise ValueError(f"Unknown training phase: {phase}")
        
        results[phase] = result
        shared_memory.set(f"training/{phase}/result", result)
    
    # Final merge and save
    final_ckpt = save_checkpoint(agent, "full", 0, results)
    print(f"\nTraining complete! Final checkpoint: {final_ckpt}")
    return results

# === Usage Example ===
if __name__ == "__main__":
    # Initialize components
    shared_memory = SharedMemory()
    agent = PerceptionAgent(config={...})  # Your agent config
    dataset = MultimodalDataset("data/training")  # Your dataset
    
    # Training configuration
    phase_configs = {
        "pretrain": {"epochs": 10, "lr": 1e-3},
        "finetune": {"epochs": 5, "lr": 5e-4},
        "rlhf": {"epochs": 3, "lr": 1e-5}
    }
    
    # Run training
    results = train_slaim(phase_configs, agent, dataset, shared_memory)
