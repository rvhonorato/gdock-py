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


def test_score_energy_weights():
    default = gdock.score(RECEPTOR_PDB, LIGAND_PDB)
    custom = gdock.score(RECEPTOR_PDB, LIGAND_PDB, w_vdw=0.0, w_elec=0.0, w_desolv=0.0)
    assert custom["total"] == 0.0
    assert default["total"] != 0.0


def test_dock():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        restraints=[(1, 1)],
        max_generations=2,
        seed=1,
    )
    assert result["generationsRun"] >= 1
    assert len(result["models"]) > 0

    model = result["models"][0]
    assert model["rank"] == 1
    assert "pdb" in model
    assert "ATOM" in model["pdb"]


def test_dock_model_fields():
    result = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=2, seed=1)
    model = result["models"][0]
    for key in ("rank", "fitness", "vdw", "elec", "desolv", "air", "pdb"):
        assert key in model
    assert isinstance(model["rank"], int)
    for key in ("fitness", "vdw", "elec", "desolv", "air"):
        assert isinstance(model[key], float)
    assert isinstance(model["pdb"], str)


def test_dock_pdb_contains_both_chains():
    result = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=2, seed=1)
    pdb = result["models"][0]["pdb"]
    chains = {line[21] for line in pdb.splitlines() if line.startswith("ATOM")}
    assert "A" in chains
    assert "B" in chains


def test_dock_models_ranked_ascending():
    result = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=5, seed=1)
    models = result["models"]
    ranks = [m["rank"] for m in models]
    assert ranks == list(range(1, len(models) + 1))


def test_dock_seed_reproducibility():
    r1 = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=3, seed=99)
    r2 = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=3, seed=99)
    assert r1["models"][0]["fitness"] == r2["models"][0]["fitness"]


def test_dock_different_seeds_differ():
    r1 = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=5, seed=1)
    r2 = gdock.dock(RECEPTOR_PDB, LIGAND_PDB, max_generations=5, seed=2)
    fitnesses1 = [m["fitness"] for m in r1["models"]]
    fitnesses2 = [m["fitness"] for m in r2["models"]]
    assert fitnesses1 != fitnesses2


def test_dock_population_size():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        max_generations=2,
        population_size=10,
        seed=1,
    )
    assert result["generationsRun"] >= 1
    assert len(result["models"]) > 0


def test_dock_sampling_returns_models_sorted_by_fitness():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        max_generations=5,
        seed=1,
        sampling=20,
    )
    models = result["models"]
    assert len(models) > 0
    fitnesses = [m["fitness"] for m in models]
    assert fitnesses == sorted(fitnesses)


def test_dock_sampling_respects_capacity():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        max_generations=3,
        seed=1,
        sampling=2,
    )
    assert len(result["models"]) <= 2


def test_dock_sampling_fills_capacity():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        max_generations=1,
        population_size=5,
        seed=1,
        sampling=5,
    )
    assert len(result["models"]) == 5


def test_dock_ncores_accepts_parameter():
    result = gdock.dock(
        RECEPTOR_PDB,
        LIGAND_PDB,
        max_generations=2,
        seed=1,
        ncores=2,
    )
    assert result["generationsRun"] >= 1
    assert len(result["models"]) > 0


def test_dock_ncores_results_match_single_core():
    kwargs = dict(
        receptor_pdb=RECEPTOR_PDB,
        ligand_pdb=LIGAND_PDB,
        max_generations=5,
        seed=42,
    )
    r1 = gdock.dock(**kwargs, ncores=1)
    r2 = gdock.dock(**kwargs, ncores=2)
    assert len(r1["models"]) == len(r2["models"])
    for m1, m2 in zip(r1["models"], r2["models"]):
        assert m1["fitness"] == m2["fitness"]


def test_dock_multicore_is_faster():
    import time

    kwargs = dict(
        receptor_pdb=RECEPTOR_PDB,
        ligand_pdb=LIGAND_PDB,
        max_generations=50,
        population_size=100,
        seed=1,
    )
    t0 = time.monotonic()
    gdock.dock(**kwargs, ncores=1)
    t_single = time.monotonic() - t0

    t0 = time.monotonic()
    gdock.dock(**kwargs, ncores=4)
    t_multi = time.monotonic() - t0

    assert t_multi < t_single, (
        f"multi-core ({t_multi:.3f}s) was not faster than single-core ({t_single:.3f}s)"
    )


def test_dock_invalid_pdb_raises():
    try:
        gdock.dock("not a pdb", LIGAND_PDB)
        assert False, "expected an error"
    except Exception:
        pass
