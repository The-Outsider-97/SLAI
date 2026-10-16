"""Apply narrowly scoped RelevanceRanker integration into the inspected SLAI v2.3 KnowledgeAgent.

Runs on a LOCAL checkout of the SLAI-v.2.3 branch. Refuses ambiguous edits.
Backs up existing file before changing. Run only once; inspect git diff.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


# Exact anchors inspected on SLAI-v.2.3. All replacements must occur exactly once.
REPLACEMENTS = [
    (
        '        self._retrieval_history_key = f"context:{self.name}"\n\n        if self.retrieval_mode in {"dense", "hybrid"}:',
        '''        self._retrieval_history_key = f"context:{self.name}"

        # Optional trained second-stage reranking. Original retrieval is always
        # available; the model never changes source or document identifiers.
        self.relevance_ranker = None
        self.relevance_ranker_enabled = bool(self.knowledge_config.get("relevance_ranker_enabled", False))
        self.relevance_ranker_pool = self._coerce_positive_int(
            self.knowledge_config.get("relevance_ranker_pool"), 100
        )
        if self.relevance_ranker_enabled:
            try:
                from .knowledge.relevance_ranker import RelevanceRanker
                path = str(self.knowledge_config.get(
                    "relevance_ranker_path", "src/agents/knowledge/models/relevance_ranker"
                ))
                model = RelevanceRanker(path, device=str(self.knowledge_config.get(
                    "relevance_ranker_device", "cpu"
                )))
                if not model.passed_baseline_gate and not bool(self.knowledge_config.get(
                    "relevance_ranker_allow_research", False
                )):
                    raise ValueError("Model failed held-out BM25 promotion gate")
                self.relevance_ranker = model
                logger.info("Knowledge RelevanceRanker activated: %s", model.artifact_sha256[:12])
            except Exception as exc:
                logger.warning("RelevanceRanker unavailable, using legacy retrieval: %s", exc)
                self.relevance_ranker = None

        if self.retrieval_mode in {"dense", "hybrid"}:''',
    ),
    (
        '            f"knowledge-agent:{self.retrieval_mode}:{k}:{len(self.knowledge_agent)}:{query}"',
        '''            f"knowledge-agent:{self.retrieval_mode}:{k}:{len(self.knowledge_agent)}:"
            f"{self.relevance_ranker.artifact_sha256 if self.relevance_ranker else 'legacy'}:{query}"''',
    ),
    (
        '        combined = self._combine_results(tfidf_results, dense_results)\n        final_results = combined[:k]',
        '''        combined = self._combine_results(tfidf_results, dense_results)
        if self.relevance_ranker is not None and len(combined) > 1:
            pool = max(k, self.relevance_ranker_pool)
            try:
                # The score tuple remains the original retrieval score; only
                # ordering changes. Neural logits are not factual confidence.
                combined = (self.relevance_ranker.rank(query, combined[:pool])
                            + combined[pool:])
            except Exception as exc:
                logger.warning("RelevanceRanker inference failed; legacy order used: %s", exc)
        final_results = combined[:k]''',
    ),
    (
        '            "dense_model_ready": self.sbert_model is not None,',
        '''            "dense_model_ready": self.sbert_model is not None,
            "relevance_ranker_active": self.relevance_ranker is not None,
            "relevance_ranker_version": (
                self.relevance_ranker.version if self.relevance_ranker else None
            ),''',
    ),
]


def apply(root: Path, dry_run: bool = False, *, verify_branch: bool = True) -> None:
    if verify_branch:
        result = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root,
            capture_output=True, text=True, check=True,
        )
        if result.stdout.strip() != "SLAI-v.2.3":
            raise RuntimeError("Refusing to modify any branch other than SLAI-v.2.3")
    target = root / "src/agents/knowledge_agent.py"
    module = root / "src/agents/knowledge/relevance_ranker.py"
    if not target.is_file() or not module.is_file():
        raise FileNotFoundError("Need SLAI v2.3 KnowledgeAgent and copied relevance_ranker.py")
    content = target.read_text(encoding="utf-8")
    updated = content
    for before, after in REPLACEMENTS:
        count = updated.count(before)
        if count != 1:
            raise ValueError(f"Integration anchor expected once, got {count}: {before[:90]!r}")
        updated = updated.replace(before, after, 1)
    compile(updated, str(target), "exec")
    if dry_run:
        print("PASS: all integration anchors match; no files changed")
        return
    backup = target.with_suffix(".py.rr_v1_backup")
    if backup.exists():
        raise FileExistsError(f"Backup already exists, refusing reapply: {backup}")
    shutil.copy2(target, backup)
    target.write_text(updated, encoding="utf-8")
    print(f"Patched {target}; backup at {backup}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    apply(a.repo_root, a.dry_run, verify_branch=True)


if __name__ == "__main__":
    main()
