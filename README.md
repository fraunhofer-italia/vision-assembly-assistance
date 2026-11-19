# Vision assembly assistance

Single-camera verification and configurable guidance for pneumatic piston assembly/disassembly, using hand tracking, object detection, finite-state machines and DTW.

## Related paper

**A Vision-Based Framework for Manual Assembly Verification and Adaptive Operator Assistance**  
Naveen Joseph Delphin, Leonardo Venturoso, Isabella Soraruf and Marco Todescato. *Manuscript draft.*

Please cite this paper when using the code. Publication details and DOI will be added when available.

## Setup

Python 3.11. Run commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[analysis]'
```

## Run

```bash
# Synthetic example, without a camera or external assets
python scripts/main.py

# Statistical analysis; requires study inputs in experiments/
python scripts/guidance_stats_final_analysis_model_B.py --output-dir results/analysis
```

Datasets and weights: [Hugging Face release index](docs/artifacts.md). See [input formats](docs/data_contract.md) and [verification](docs/verification.md). Camera/model workflows require the optional dependencies; see `scripts/main_full_pipeline.py --help`.

## License

[MIT](LICENSE) for original project code. Third-party dependencies and models retain their own [licenses](docs/licensing.md).
