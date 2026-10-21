# -*- coding: utf-8 -*-
"""Regenerate the 18 case-study catalyst conformers exactly as the deployment
script does, and write them to a single SDF.

Protocol (baselines/server_package/server_case_study.py, smiles_to_3d_graph):

    mol   = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    rc    = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=200)
    mmff  = AllChem.MMFFOptimizeMolecule(mol_h)   # often -1: no MMFF params for TM
    mol2  = Chem.RemoveHs(mol_h)                  # <- conformer used for the graph

The SDF stores mol2 (heavy atoms only, ETKDG coordinates) with per-record
properties name/smiles/embed_rc/mmff_rc, plus a JSON sidecar with the return
codes and the RDKit version used.  The CANDIDATES table is read from the
deployment script itself (ast.literal_eval), not retyped.
"""
import ast
import json
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data_indices')
SRC = r'D:\ml\tmqm_GAT\baselines\server_package\server_case_study.py'


def load_candidates(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'CANDIDATES':
                    return ast.literal_eval(node.value)
    raise RuntimeError('CANDIDATES not found')


def main():
    os.makedirs(OUT, exist_ok=True)
    cand = load_candidates(SRC)
    print(f'candidates: {len(cand)}', flush=True)

    writer = Chem.SDWriter(os.path.join(OUT, 'catalysts_18_etkdg.sdf'))
    writer.SetKekulize(True)
    meta = []
    n_fail = 0
    for name, smi in cand.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            meta.append({'name': name, 'smiles': smi, 'error': 'MolFromSmiles'})
            n_fail += 1
            print(f'  {name}: parse failed', flush=True)
            continue
        mol_h = Chem.AddHs(mol)
        rc = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=200)
        if rc != 0:
            meta.append({'name': name, 'smiles': smi, 'error': f'EmbedMolecule rc={rc}'})
            n_fail += 1
            print(f'  {name}: embed failed rc={rc}', flush=True)
            continue
        mmff = AllChem.MMFFOptimizeMolecule(mol_h)
        mol2 = Chem.RemoveHs(mol_h)
        mol2.SetProp('_Name', name)
        mol2.SetProp('name', name)
        mol2.SetProp('smiles', smi)
        mol2.SetProp('embed_rc', str(rc))
        mol2.SetProp('mmff_rc', str(mmff))
        mol2.SetProp('rdkit_version', Chem.rdBase.rdkitVersion)
        writer.write(mol2)

        conf = mol2.GetConformer()
        pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(mol2.GetNumAtoms())])
        meta.append({
            'name': name, 'smiles': smi,
            'canonical_smiles': Chem.MolToSmiles(mol2),
            'formula': Chem.rdMolDescriptors.CalcMolFormula(mol2),
            'n_heavy_atoms': mol2.GetNumAtoms(), 'n_atoms_with_h': mol_h.GetNumAtoms(),
            'embed_rc': rc, 'mmff_rc': mmff,
            'coord_min': pos.min(axis=0).tolist(), 'coord_max': pos.max(axis=0).tolist(),
        })
        print(f'  {name}: atoms={mol2.GetNumAtoms()}, embed={rc}, mmff={mmff}', flush=True)
    writer.close()

    n_records_missing_h = sum(1 for m in meta if 'n_heavy_atoms' in m and m['n_heavy_atoms'] != m['n_atoms_with_h'])
    recon = {
        'source': SRC,
        'rdkit_version': Chem.rdBase.rdkitVersion,
        'protocol': 'AddHs -> EmbedMolecule(randomSeed=42, maxAttempts=200) -> '
                    'MMFFOptimizeMolecule -> RemoveHs',
        'n_candidates': len(cand),
        'n_written': sum(1 for m in meta if 'n_heavy_atoms' in m),
        'n_failures': n_fail,
        'mmff_return_codes': sorted({m.get('mmff_rc') for m in meta if 'mmff_rc' in m}),
        'molecules': meta,
    }
    with open(os.path.join(OUT, 'catalysts_18_etkdg.json'), 'w', encoding='utf-8') as f:
        json.dump(recon, f, ensure_ascii=False, indent=2)
    print(f'wrote catalysts_18_etkdg.sdf ({recon["n_written"]} records) and .json; '
          f'MMFF return codes={recon["mmff_return_codes"]}', flush=True)


if __name__ == '__main__':
    main()
