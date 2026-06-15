# gdock-py

Minimal Python bindings for [gdock](https://gdock.org), an information-driven
protein-protein docking engine using a genetic algorithm.

## Installation

```bash
pip install gdock
```

## Usage

```python
import gdock

receptor_pdb = open("receptor.pdb").read()
ligand_pdb = open("ligand.pdb").read()

# Score a receptor-ligand complex
scores = gdock.score(receptor_pdb, ligand_pdb)
print(scores)  # {"vdw": ..., "elec": ..., "desolv": ..., "total": ...}

# Run the full GA docking pipeline
result = gdock.dock(
    receptor_pdb,
    ligand_pdb,
    restraints=[(10, 20), (15, 25)],  # (receptor_resseq, ligand_resseq) pairs
    max_generations=250,
    seed=42,
)
print(f"Generations run: {result['generationsRun']}")
print(f"Best fitness: {result['poses'][0]['fitness']}")
best_pdb = result["poses"][0]["pdb"]
```

## API

### `score(receptor_pdb, ligand_pdb, w_vdw=0.4, w_elec=0.05, w_desolv=3.4) -> dict`

Computes VDW, electrostatic and desolvation energy terms for a receptor-ligand
pose. Returns `{"vdw", "elec", "desolv", "total"}`.

### `dock(receptor_pdb, ligand_pdb, restraints=None, max_generations=250, seed=42) -> dict`

Runs the genetic algorithm docking pipeline and returns ranked poses:

```
{
  "generationsRun": int,
  "poses": [
    {"rank", "fitness", "vdw", "elec", "desolv", "air", "pdb"},
    ...
  ]
}
```

## Development

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install maturin pytest
maturin develop
pytest tests/
```

## License

0BSD
