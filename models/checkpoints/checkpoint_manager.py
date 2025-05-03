import os
import json
import numpy as np
import datetime
import torch


class CheckpointManager:
    def __init__(self, base_dir="models/checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, model, tokenizer, metadata=None, version=None, format="npz"):
        if format not in ["npz", "torch"]:
            raise ValueError("Unsupported format. Use 'npz' or 'torch'.")
        
        if format == "npz":
            self.save_npz(model, tokenizer, metadata, version)
        else:
            self.save_torch(model, tokenizer, metadata, version)

    def load(self, model, tokenizer, version, format="npz"):
        if format not in ["npz", "torch"]:
            raise ValueError("Unsupported format. Use 'npz' or 'torch'.")
        
        if format == "npz":
            self.load_npz(model, tokenizer, version)
        else:
            self.load_torch(model, tokenizer, version)

    def save_npz(self, model, tokenizer, metadata=None, version=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = version or f"v_{timestamp}"
        save_path = os.path.join(self.base_dir, version_tag)
        os.makedirs(save_path, exist_ok=True)

        model_weights = {
            name: param.data if hasattr(param, 'data') else param
            for name, param in model.named_parameters()
        }
        np.savez(os.path.join(save_path, "model_weights.npz"), **model_weights)

        with open(os.path.join(save_path, "tokenizer_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.word_to_id, f, ensure_ascii=False, indent=2)

        if metadata:
            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"NPZ checkpoint saved at {save_path}")

    def save_torch(self, model, tokenizer, metadata=None, version=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = version or f"v_{timestamp}"
        save_path = os.path.join(self.base_dir, version_tag)
        os.makedirs(save_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_path, "model_weights.pt"))

        with open(os.path.join(save_path, "tokenizer_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.word_to_id, f, ensure_ascii=False, indent=2)

        if metadata:
            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"PyTorch checkpoint saved at {save_path}")

    def load_npz(self, model, tokenizer, version):
        load_path = os.path.join(self.base_dir, version)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint {version} not found at {load_path}.")

        weights = np.load(os.path.join(load_path, "model_weights.npz"))
        for name, param in model.named_parameters():
            if name in weights:
                param.data = weights[name]
            else:
                print(f"Warning: Weight {name} not found in checkpoint.")

        with open(os.path.join(load_path, "tokenizer_vocab.json"), "r", encoding="utf-8") as f:
            tokenizer.word_to_id = json.load(f)
            tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}
            tokenizer.vocab_size = len(tokenizer.word_to_id)

        print(f"NPZ checkpoint {version} loaded successfully")

    def load_torch(self, model, tokenizer, version):
        load_path = os.path.join(self.base_dir, version)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint {version} not found at {load_path}.")

        model.load_state_dict(torch.load(os.path.join(load_path, "model_weights.pt")))

        with open(os.path.join(load_path, "tokenizer_vocab.json"), "r", encoding="utf-8") as f:
            tokenizer.word_to_id = json.load(f)
            tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}
            tokenizer.vocab_size = len(tokenizer.word_to_id)

        print(f"PyTorch checkpoint {version} loaded successfully")

    def list_checkpoints(self):
        versions = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        return sorted(versions)

    def delete_checkpoint(self, version):
        delete_path = os.path.join(self.base_dir, version)
        if os.path.exists(delete_path):
            for root, dirs, files in os.walk(delete_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(delete_path)
            print(f"Deleted checkpoint {version}")
        else:
            print(f"Checkpoint {version} not found.")
