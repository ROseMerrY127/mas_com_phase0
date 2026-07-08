# Phase0 Identity-Router GSM8K Experiment

This project implements the Phase0 plan from `routing_mechanism_spec_final_1.md`:
identity routing over a fixed math MAS topology, full pass-through edge logging, and a
single-agent CoT baseline on local GSM8K data.

## Data

Default input:

```text
gsm8K/test-00000-of-00001.parquet
```

The loader supports parquet, JSONL, and JSON records with `question` and `answer` fields.

## Configuration

Edit `config/phase0.yaml` for experiment defaults. Model credentials are read from the
environment or `.env`:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=...        # optional for OpenAI-compatible gateways
OPENAI_MODEL=...           # optional override for config model
```

## Run

```powershell
python run_phase0.py
```

Useful smoke run:

```powershell
python run_phase0.py --sample-size 1
```

Each run writes:

```text
runs/phase0_<timestamp>/traces.jsonl
runs/phase0_<timestamp>/predictions.jsonl
runs/phase0_<timestamp>/summary.json
```

## Local Checks

```powershell
python tests\test_phase0.py
python -m compileall src tests run_phase0.py
```
