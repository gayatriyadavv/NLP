# Real 3D structure sources

All coordinates are experimental structures from the Protein Data Bank (wwPDB data is
released under CC0 1.0). The files were obtained from test-data directories of two open-source
packages because rcsb.org was not reachable from the build environment:

* **MDAnalysisTests 2.10.0** (PyPI sdist, `MDAnalysisTests/data/`, GPL/LGPL package; the PDB
  coordinates themselves are wwPDB data).
* **Biopython** (`Tests/PDB/` on GitHub master, Biopython License).

| File | PDB ID | Molecule | Source package |
|---|---|---|---|
| 19hc.pdb.gz | 19HC | Nine-haem cytochrome c | MDAnalysisTests |
| 1a28.pdb.gz | 1A28 | Progesterone receptor ligand-binding domain | MDAnalysisTests |
| 1grm_single.pdb | 1GRM | Gramicidin A (peptide antibiotic) | MDAnalysisTests |
| 1hvr.pdb | 1HVR | HIV-1 protease + inhibitor | MDAnalysisTests |
| 1osm.pdb.gz | 1OSM | OmpK36 outer membrane porin | MDAnalysisTests |
| 4E43.pdb | 4E43 | HIV protease | MDAnalysisTests |
| 5a7u.pdb | 5A7U | ADR1 regulatory protein zinc finger | MDAnalysisTests |
| adk_closed.pdb | 1AKE | Adenylate kinase, closed state | MDAnalysisTests |
| adk_open.pdb | 4AKE | Adenylate kinase, open state | MDAnalysisTests |
| cobrotoxin.pdb | 1V6P | Cobrotoxin (first frame, solvent stripped on load) | MDAnalysisTests |
| nmr_neopetrosiamide.pdb | 2JUY | Neopetrosiamide A (NMR, model 1 used) | MDAnalysisTests |
| fullerene.pdb.gz | — | C60 fullerene (small molecule, all atoms used) | MDAnalysisTests |
| 1A8O.pdb | 1A8O | HIV capsid C-terminal domain | Biopython |
| 1LCD.pdb | 1LCD | Lac repressor headpiece–DNA complex | Biopython |
| 2BEG.pdb | 2BEG | Amyloid-β(1-42) fibril | Biopython |
| 2XHE.pdb | 2XHE | Unc18 / syntaxin complex | Biopython |

Proteins are represented by their Cα trace (first model, all chains). Molecules without Cα
atoms (fullerene) use all heavy atoms.
