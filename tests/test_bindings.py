import gdock

RECEPTOR_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)

LIGAND_PDB = (
    "ATOM      1  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2  CA  ALA B   2      13.800   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)


def test_score():
    result = gdock.score(RECEPTOR_PDB, LIGAND_PDB)
    assert set(result.keys()) == {"vdw", "elec", "desolv", "total"}
    for value in result.values():
        assert isinstance(value, float)


def test_dock():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        restraints=[(1, 1)],
        max_generations=2,
        seed=1,
    )
    assert result["generationsRun"] >= 1
    assert len(result["poses"]) > 0

    pose = result["poses"][0]
    assert pose["rank"] == 1
    assert "pdb" in pose
    assert "ATOM" in pose["pdb"]
