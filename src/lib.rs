//! Minimal Python bindings for `gdock` - information-driven protein-protein docking.

use gdock::chromosome::Chromosome;
use gdock::constants::{self, EnergyWeights};
use gdock::fitness;
use gdock::population::Population;
use gdock::restraints::create_restraints_from_pairs;
use gdock::runner::{run_ga, select_models};
use gdock::structure::{self, Molecule};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

/// Parse a single-model PDB string into a `Molecule`, erroring if it has no atoms.
fn parse_molecule(pdb: &str, label: &str) -> PyResult<Molecule> {
    structure::read_pdb_from_str(pdb)
        .0
        .into_iter()
        .next()
        .ok_or_else(|| PyValueError::new_err(format!("{label} PDB contains no atoms")))
}

/// Score a receptor-ligand complex using the gdock energy function.
///
/// Args:
///     receptor_pdb: PDB string of the receptor
///     ligand_pdb: PDB string of the ligand
///     w_vdw: Weight for van der Waals energy (default: 0.4)
///     w_elec: Weight for electrostatic energy (default: 0.05)
///     w_desolv: Weight for desolvation energy (default: 3.4)
///
/// Returns:
///     dict with keys: vdw, elec, desolv, total
#[pyfunction]
#[pyo3(signature = (receptor_pdb, ligand_pdb, w_vdw=constants::DEFAULT_W_VDW, w_elec=constants::DEFAULT_W_ELEC, w_desolv=constants::DEFAULT_W_DESOLV))]
fn score(
    py: Python<'_>,
    receptor_pdb: &str,
    ligand_pdb: &str,
    w_vdw: f64,
    w_elec: f64,
    w_desolv: f64,
) -> PyResult<Py<PyDict>> {
    let receptor = parse_molecule(receptor_pdb, "Receptor")?;
    let ligand = parse_molecule(ligand_pdb, "Ligand")?;

    let vdw = fitness::vdw_energy(&receptor, &ligand);
    let elec = fitness::elec_energy(&receptor, &ligand);
    let desolv = fitness::desolv_energy(&receptor, &ligand);
    let total = w_vdw * vdw + w_elec * elec + w_desolv * desolv;

    let dict = PyDict::new(py);
    dict.set_item("vdw", vdw)?;
    dict.set_item("elec", elec)?;
    dict.set_item("desolv", desolv)?;
    dict.set_item("total", total)?;
    Ok(dict.into())
}

/// A single ranked docking pose.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Pose {
    rank: usize,
    fitness: f64,
    vdw: f64,
    elec: f64,
    desolv: f64,
    air: f64,
    pdb: String,
}

/// Result of [`dock`].
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DockResult {
    generations_run: u64,
    poses: Vec<Pose>,
}

/// Run the genetic algorithm docking pipeline.
///
/// Args:
///     receptor_pdb: PDB string of the receptor
///     ligand_pdb: PDB string of the ligand
///     restraints: List of (receptor_resseq, ligand_resseq) pairs for distance restraints
///     max_generations: Maximum number of GA generations (default: 250)
///     seed: Random seed for reproducibility (default: 42)
///
/// Returns:
///     dict with keys:
///         - generations_run: number of generations executed
///         - poses: ranked poses, each with rank, fitness, vdw, elec, desolv, air, pdb
#[pyfunction]
#[pyo3(signature = (receptor_pdb, ligand_pdb, restraints=None, max_generations=None, seed=None))]
fn dock(
    py: Python<'_>,
    receptor_pdb: &str,
    ligand_pdb: &str,
    restraints: Option<Vec<(i32, i32)>>,
    max_generations: Option<u64>,
    seed: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let receptor = parse_molecule(receptor_pdb, "Receptor")?;
    let ligand = parse_molecule(ligand_pdb, "Ligand")?;

    let pairs = restraints.unwrap_or_default();
    let restraint_list = create_restraints_from_pairs(&receptor, &ligand, &pairs);

    let max_generations = max_generations.unwrap_or(constants::MAX_GENERATIONS);
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(constants::RANDOM_SEED));

    let mut chromosomes = Vec::with_capacity(constants::POPULATION_SIZE as usize);
    for _ in 0..constants::POPULATION_SIZE {
        chromosomes.push(Chromosome::new(&mut rng));
    }

    let pop = Population::new(
        chromosomes,
        receptor.clone(),
        ligand.clone(),
        Molecule::new(),
        restraint_list,
        EnergyWeights::default(),
        None,
    );

    let ga_result = run_ga(pop, &mut rng, max_generations, |_, _| {});
    let hof_entries = ga_result.hall_of_fame.entries();
    let selected = select_models(hof_entries, &receptor, &ligand);

    let poses: Vec<Pose> = selected
        .ranked
        .iter()
        .enumerate()
        .map(|(i, &hof_idx)| {
            let entry = &hof_entries[hof_idx];
            let docked = ligand
                .clone()
                .rotate(entry.genes[0], entry.genes[1], entry.genes[2])
                .displace(entry.genes[3], entry.genes[4], entry.genes[5]);
            Pose {
                rank: i + 1,
                fitness: entry.fitness,
                vdw: entry.vdw,
                elec: entry.elec,
                desolv: entry.desolv,
                air: entry.air,
                pdb: docked.to_pdb_string(),
            }
        })
        .collect();

    let result = DockResult {
        generations_run: ga_result.generations_run,
        poses,
    };

    let json = serde_json::to_string(&result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize result: {e}")))?;
    let loads = py.import("json")?.getattr("loads")?;
    loads.call1((json,))?.extract()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(dock, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    use pyo3::Python;

    #[test]
    fn test_score() {
        let receptor_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C\nEND\n";
        let ligand_pdb =
            "ATOM      1  CA  ALA B   1       5.000   0.000   0.000  1.00  0.00           C\nEND\n";

        Python::attach(|py| {
            let result = score(py, receptor_pdb, ligand_pdb, 0.4, 0.05, 3.4).unwrap();
            let dict = result.bind(py);
            let vdw: f64 = dict.get_item("vdw").unwrap().unwrap().extract().unwrap();
            let total: f64 = dict.get_item("total").unwrap().unwrap().extract().unwrap();
            assert!(vdw.is_finite());
            assert!(total.is_finite());
        });
    }

    #[test]
    fn test_dock_runs_a_few_generations() {
        let receptor_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\nEND\n";
        let ligand_pdb = "ATOM      1  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C\nATOM      2  CA  ALA B   2      13.800   0.000   0.000  1.00  0.00           C\nEND\n";

        Python::attach(|py| {
            let result = dock(
                py,
                receptor_pdb,
                ligand_pdb,
                Some(vec![(1, 1)]),
                Some(2),
                Some(1),
            )
            .unwrap();
            let dict = result.bind(py);
            let generations_run: u64 = dict
                .get_item("generationsRun")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(generations_run >= 1);
            let poses = dict.get_item("poses").unwrap().unwrap();
            let poses = poses.downcast::<PyList>().unwrap();
            assert!(poses.len() > 0);
        });
    }
}
