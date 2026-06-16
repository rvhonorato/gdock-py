"""Smoke test for the published `gdock` package on PyPI.

Run in a fresh virtual environment, e.g.:

    uv venv .venv-example
    source .venv-example/bin/activate
    uv pip install gdock
    python examples/run-example.py
"""

from pathlib import Path

import gdock

DATA_DIR = Path(__file__).parent / "data"
RECEPTOR_PDB = DATA_DIR / "2oob_A.pdb"
LIGAND_PDB = DATA_DIR / "2oob_B.pdb"

RESTRAINTS = [
    (933, 6),
    (936, 8),
    (940, 42),
    (941, 44),
    (946, 45),
    (950, 46),
]


def main():
    print(f"gdock version: {gdock.__version__}")

    receptor_pdb = open(RECEPTOR_PDB).read()
    ligand_pdb = open(LIGAND_PDB).read()

    print("\n== score() ==")
    scores = gdock.score(receptor_pdb, ligand_pdb)
    print(scores)

    print("\n== dock() ==")
    result = gdock.dock(
        receptor_pdb,
        ligand_pdb,
        restraints=RESTRAINTS,
        max_generations=10,
        population_size=50,
        ncores=4,
        seed=42,
    )
    print(f"generations run: {result['generationsRun']}")
    for model in result["models"]:
        print(
            f"  rank={model['rank']} fitness={model['fitness']:.3f} "
            f"vdw={model['vdw']:.3f} elec={model['elec']:.3f} "
            f"desolv={model['desolv']:.3f} air={model['air']:.3f}"
        )

    best_pdb = result["models"][0]["pdb"]
    with open("best_model.pdb", "w") as f:
        f.write(best_pdb)
    print("\nWrote best model to best_model.pdb")

    print("\n== dock() with sampling ==")
    result = gdock.dock(
        receptor_pdb,
        ligand_pdb,
        restraints=RESTRAINTS,
        max_generations=10,
        population_size=50,
        seed=42,
        ncores=4,
        sampling=500,
    )
    print(f"generations run: {result['generationsRun']}")
    print(f"models returned: {len(result['models'])}")

    sampling_dir = Path("sampling")
    sampling_dir.mkdir(exist_ok=True)
    for model in result["models"]:
        pdb_path = sampling_dir / f"gdock_{model['rank']}.pdb"
        pdb_path.write_text(model["pdb"])
    print(f"  wrote {len(result['models'])} models to {sampling_dir}/")


if __name__ == "__main__":
    main()
