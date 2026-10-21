# Large files: inventory and hosting plan

The files below are referenced by the manuscript but are too large for the git
history. They are distributed as assets of the `v1.0-review` GitHub release;
the repository is public at submission. After acceptance the same files will be
archived in a DOI-bearing record. None of them is committed to the repository.

Checksums are md5, matching the rest of the project's records.

## Model weights

| file | bytes | members | md5 | role |
|---|---:|---:|---|---|
| `ensemble_checkpoint_A_8d6d.pt` | 47,247,385 | 9 | `1a286749d0413183136a4543af0b742a` | variant A, SMILES-only (8d/6d) |
| `ensemble_checkpoint_B_8d22d.pt` | 47,285,511 | 9 | `964849ac214b36304a82b42c73edb510` | variant B, 3D deployment (8d/22d, no WBO) |
| `ensemble_checkpoint_C_11d6d.pt` | 47,292,359 | 9 | `cf4f3474c980d1980d930b0beb935524` | variant C, 11d/6d |
| `ensemble_checkpoint_D_11d23d.pt` | 47,330,229 | 9 | `479d3b58c8d30460a430bf33ad2a46db` | variant D, full 3D (11d/23d) |
| `ensemble_checkpoint_SchNet.pt` | 96,122,164 | 9 | `15661382cb757bc25ed7019893d9cd60` | SchNet baseline ensemble |
| DimeNet++ three-member weights | — | 3 | — | **not on this machine**; produced on the server by `server_dimenet_fixed.py`, must be copied back before the record is frozen |

Total of the five files present locally: 285,277,648 bytes (272 MiB).

## DFT reference logs

Twelve Gaussian 16 C.01 logs, 213,934,099 bytes in total, in `D:\ml\log`.
Each file is a geometry optimisation (TPSSh/def2-SVP) followed by a single point
with full population analysis at the same level, as described in the SI.

They are distributed as `dft_logs_12.zip` (43,692,300 bytes, md5
`cfd424c2970c76ffbc1141a5f0d69d1d`), built with fixed timestamps. The archive was
re-read after compression: CRC and the md5 of every member match the table below.

| case-study name | log file | formula (Gaussian stoichiometry) | bytes | md5 |
|---|---|---|---:|---|
| Zr-cat3 | `Zr-cat3-2.log` | C10H10I2Zr | 3,244,512 | `0fd1f5542d0586c5dcadfcfae819db50` |
| Co-cat1 | `Co-cat1-3.log` | C4H6CoO4 | 2,230,608 | `c873dc2d40370c11616acd4c9cb2f503` |
| Fe-cat1 | `Fe-cat1-2.log` | C9H18Cl3FeNO | 6,409,854 | `5ea945794ce62dd1879c4c3de0c11e69` |
| Ni-cat3 | `Ni-cat3-2.log` | C26H14F6IN2Ni | 19,766,658 | `6312fd9ff344a48bdbd315a37d1edc5e` |
| Pd-cat2 | `Pd-cat1-2.log` | C38H24F4O4P2Pd | 21,230,499 | `55a26c5e5decfedcff46ce657d6500e5` |
| cis-PdCl2(PPh3)2 | `cis-PdCl2-PPh32.log` | C36H30Cl2P2Pd | 20,511,583 | `fd2e70183da61d95a58171209347de10` |
| trans-PdCl2(PPh3)2 | `PdCl2-PPh32.log` | C36H30Cl2P2Pd | 19,160,646 | `76a11e91b5a690b1499529b7e57da075` |
| Rh(CO)2(acac) | `Rh-acac-CO2-1.log` | C7H7O4Rh | 2,386,261 | `fab9418ccbb4545940e14bb3fb661953` |
| Ni(IMes)2Cl2 | `Ni-IMes2Cl2_m1.log` | C42H48Cl2N4Ni | 32,660,367 | `def2c70474de8e1c99e4e3d951fab21f` |
| Pt(dppe)Cl2 | `Pt-dppe-Cl2-1.log` | C26H24Cl2P2Pt | 11,725,679 | `d7a53bef23480b27d6600f52a59a881a` |
| — (alternate) | `Ni-IMes2Cl2_m3.log` | C42H48Cl2N4Ni | 56,546,810 | `ba74a29dedaa9d7c8b1cbb6294404e61` |
| — (extra) | `Ni-NHC-Cl.log` | C21H25Cl2N2Ni | 18,060,622 | `ecf3b9eecafdcfc1ba1cc6dfea834952` |

Notes for the record description:

* The file `Pd-cat1-2.log` has a misleading name. Its stoichiometry
  (C38H24F4O4P2Pd) and the four fluorine atoms match **Pd-cat2**, the
  difluorodioxolane-diphosphine complex; `Pd-cat1` has no fluorine.
* `Ni-IMes2Cl2_m1.log` and `_m3.log` are the singlet and triplet solutions of the
  same complex. The 10-reference table uses one of them; both are included so the
  choice is visible. `m1` lies 0.021 Ha lower at the final SCF.
* `Ni-NHC-Cl.log` is an early Ni-NHC calculation that is not one of the ten
  reference complexes. It is included in the released twelve-file set so that
  the full calculation record is available.

## Domain tensor (optional)

| file | bytes | md5 | note |
|---|---:|---|---|
| `new_domain_iso.pt` | 1,107,170,889 | `3035057aeb8638680d676283ace2194b` | 90,377 graphs, labels, `csv_idx`, `te_idx`. Optional: `retained_ids.csv` + `atom_maps.npz` already let a reader rebuild the indexing; include the tensor only if hosting allows. |
