# SLAI Evaluation Agent: TestGenerator (experimental)

**Not production-ready. Disabled by default. No generated code is executed.**

This feature adds a dedicated conditional byte-level GRU encoder/decoder,
trained on source-to-test pairs. It is a real trainable generative model,
not a set of static test templates. The existing EvaluationAgent retains
orchestration ownership. Existing evaluator modules remain intact.

## Training data

Input is JSONL. Every example must include source (Python code), test (pytest
source), target_symbol, group, provenance, license, and verified=true.
Optional: target_module, requirements, verified_failure, synthetic.
Group means independent project or module family; all examples from a
group stay in one partition. Only verified examples are accepted.
Syntactic screening does not prove that a label is verified. Dataset owners
must review rights, licensing, secrets and behavioral expectations.

Example record (as one JSONL line):

{"source":"def square(x):\n    return x*x\n","test":"from example import square\n\ndef test_square():\n    assert square(0) == 0\n","target_symbol":"square","target_module":"example","group":"project-A","provenance":"reviewed source and test commit","license":"owner-authorized","verified":true,"synthetic":false}

At least 10 unique examples from 5 independent groups are required for a
smoke dataset preparation; this is NOT sufficient evidence for production.

## Windows PowerShell

From the SLAI root:

    $python = "G:\GAIA\AI\SLAI\venv\Scripts\python.exe"
    $module = "src.agents.evaluators.modules.test_generator"
    $inputData = ".\data\training\evaluation\test_generator\verified_pairs.jsonl"
    $prepared = ".\data\training\evaluation\test_generator\prepared"
    $checkpoint = ".\data\training\evaluation\test_generator\run_001"

    & $python -m $module prepare --input $inputData --output $prepared
    & $python -m $module train --prepared $prepared --checkpoint $checkpoint --epochs 3 --seed 42
    & $python -m $module evaluate --prepared $prepared --checkpoint $checkpoint
    & $python -m pytest tests/agents/test_evaluation_test_generator.py -q

The trainer records seed, split hashes/counts and validation loss. The
checkpoint is a PyTorch state dictionary loaded with weights_only=True,
strict model compatibility checks, and SHA256 integrity verification.
It starts with promotion=experimental and is NOT automatically approved.
Train/test leakage and near-duplicate examples must be reviewed beyond the
current exact-pair deduplicator.

## Agent integration

The agent-level configuration is in
src/agents/base/configs/agents_config.yaml:

    evaluation_agent:
      test_generator:
        mode: shadow
        checkpoint_dir: "data/training/evaluation/test_generator/run_001"

Call the existing agent with:

    agent.predict({
        "operation": "generate_tests",
        "source": "def square(x):\n    return x*x\n",
        "target_symbol": "square",
        "target_module": "example",
        "category": "boundary",
        "framework": "pytest"
    })

Supported request category labels include unit, boundary, regression,
integration, property, failure_driven, parameterized and improvement.
These labels DO NOT establish trained competence for each task type.

Default mode disabled preserves existing EvaluationAgent behavior.
Shadow mode generates diagnostics without application.
Advisory/active reject experimental checkpoints.
Even when an approved checkpoint is configured, these modes still return
review-only candidates: no automatic execution or repository changes.

## Security and scientific limitations

Generated code is untrusted. The AST scanner rejects selected dangerous
imports and calls, syntax failures, trivial assertions, and missing target
references. This scan is NOT a sandbox, cannot prove absence of malicious
behavior, and does not validate the expected oracle. A passing generated
test may still be meaningless.

The system does not execute generated code or invoke pytest on generated
candidates, because a safe Windows execution boundary has NOT been
independently verified. No legitimate training dataset, production-grade
checkpoint, measured coverage gains, mutation-score improvement, end-to-end
Windows sandbox evidence, or production-readiness certification is included
with this change.

Further requirements: independently verified training pairs, project-held-out
benchmarks, model-vs-baseline comparisons, mutation and regression checks,
sandboxed execution authorization, comprehensive subsystem regression tests,
and a reviewed promotion/rollback process. Compare the initial GRU with
a LANTRA fine-tuned model before selecting a production generator.
