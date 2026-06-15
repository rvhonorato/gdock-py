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

# Receptor:ligand residue restraint pairs (from the gdock CLAUDE.md example)
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
        seed=42,
    )
    print(f"generations run: {result['generationsRun']}")
    for pose in result["poses"]:
        print(
            f"  rank={pose['rank']} fitness={pose['fitness']:.3f} "
            f"vdw={pose['vdw']:.3f} elec={pose['elec']:.3f} "
            f"desolv={pose['desolv']:.3f} air={pose['air']:.3f}"
        )

    best_pdb = result["poses"][0]["pdb"]
    with open("best_pose.pdb", "w") as f:
        f.write(best_pdb)
    print("\nWrote best pose to best_pose.pdb")


if __name__ == "__main__":
    main()
