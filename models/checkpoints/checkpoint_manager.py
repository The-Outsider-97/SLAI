import os
import json
import torch
import shutil
import hashlib
import datetime
import numpy as np


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
        if version is None:
            raise ValueError("Checkpoint version cannot be None")
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

        if hasattr(model, 'parameters'):
            model_weights = {
                name: param.data.cpu().numpy()
                for name, param in model.named_parameters()
            }
        else:
            model_weights = {
                name: param.data
                for name, param in model.transformer.parameters()
            }

        np.savez(os.path.join(save_path, "model_weights.npz"), **model_weights)

        with open(os.path.join(save_path, "tokenizer_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.word_to_id, f, ensure_ascii=False, indent=2)

        if metadata:
            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"NPZ checkpoint saved at {save_path}")

        archive_name = os.path.join(self.base_dir, f"{version_tag}.tar.gz")
        shutil.make_archive(
            base_name=archive_name.replace('.tar.gz', ''),
            format='gztar',
            root_dir=save_path
        )
        print(f"Compressed archive created at {archive_name}")

    def save_torch(self, model, tokenizer, optimizer, current_epoch, metadata=None, version=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = version or f"v_{timestamp}"
        save_path = os.path.join(self.base_dir, version_tag)
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(save_path, "model_weights.pt")
        )

        # === Precision handling ===
        if hasattr(model, 'precision_mode'):
            if model.precision_mode == 'fp16':
                model = model.half()
            elif model.precision_mode == 'int8':
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
    
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': current_epoch,
            'rng_state': torch.get_rng_state()
        }, os.path.join(save_path, "model_weights.pt"))
    
        with open(os.path.join(save_path, "tokenizer_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.word_to_id, f, ensure_ascii=False, indent=2)
    
        if metadata:
            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
        print(f"PyTorch checkpoint saved at {save_path}")

        archive_name = os.path.join(self.base_dir, f"{version_tag}.tar.gz")
        shutil.make_archive(
            base_name=archive_name.replace('.tar.gz', ''),
            format='gztar',
            root_dir=save_path
        )
        print(f"Compressed archive created at {archive_name}")

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

    def load_torch(self, model, tokenizer, metadata=None, version=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = version or f"v_{timestamp}"
        save_path = os.path.join(self.base_dir, version_tag)
        os.makedirs(save_path, exist_ok=True)
        load_path = os.path.join(self.base_dir, version)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint {version} not found at {load_path}.")
    
        weight_file = os.path.join(load_path, "model_weights.pt")
    
        def compute_sha256(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        
        weight_file = os.path.join(save_path, "model_weights.pt")
        hash_value = compute_sha256(weight_file)
        with open(weight_file + ".sha256", "w") as f:
            f.write(hash_value) 

        model.load_state_dict(
            torch.load(weight_file, map_location=torch.device('cpu'))
        )

        with open(os.path.join(load_path, "tokenizer_vocab.json"), "r", encoding="utf-8") as f:
            tokenizer.word_to_id = json.load(f)
            tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}
            tokenizer.vocab_size = len(tokenizer.word_to_id)
    
        print(f"PyTorch checkpoint {version} loaded successfully")

    def list_checkpoints(self):
        # Only list directories starting with 'v_' or 'initial'
        versions = [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d)) 
            and (d.startswith("v_") or d.startswith("initial"))
        ]
        return sorted(versions)
    
    def get_latest_checkpoint(self):
        checkpoints = [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        if not checkpoints:
            return None
        return sorted(checkpoints)[-1]  # Assumes list_checkpoints() returns sorted versions

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
