# Data provenance: the 90,377-complex domain

This note records the source files, checksums, and filtering chain for the
90,377-complex domain used in the manuscript. It complements
`data_indices/index_reconciliation.json` and the row-level tables in
`data_indices/`. The older `repro/数据版本说明.md` describes the legacy
90,317 domain and is retained unchanged as a historical record.

## Source release

tmQM, 13 August 2024 release. The files below are the copies actually read by
the data-processing scripts on the local machine
(`data_tmQM/local_vB/`). Byte counts are exact; md5 values were recomputed on
2026-10-21 and agree with `data_indices/index_reconciliation.json`.

| file | bytes | md5 |
|---|---:|---|
| `tmQM_y.csv` | 19,925,955 | `f8eeb967b71938f1a8c1d44be8f1f043` |
| `tmQM_X1.xyz.gz` | 68,010,672 | `f273275f288b375ab20b2f9d5c5c227f` |
| `tmQM_X2.xyz.gz` | 68,016,824 | `563be6efff4d99b64e3590caeb5c5699` |
| `tmQM_X3.xyz.gz` | 67,924,252 | `c4a9f3c48c47792086c0532d31eb6d6a` |
| `tmQM_X1.BO.gz` | 30,430,198 | `cf64d4f145f649bc68c18e6f8704c8d1` |
| `tmQM_X2.BO.gz` | 30,901,635 | `00b561bf31e2c967e3d919059701b936` |
| `tmQM_X3.BO.gz` | 30,193,825 | `ea4bd75955d596224145b8e4cf4ebbe5` |

## Filtering chain

| step | count | rule |
|---|---:|---|
| source records | 108,541 | all rows in `tmQM_y.csv` |
| contains at least one transition metal | 108,116 | 425 records without a transition metal excluded (`no_TM`) |
| parseable SMILES in the training stack | 100,733 | 7,383 transition-metal records with empty or unparseable SMILES excluded (`unparseable_SMILES`) |
| element-preserving graph match | 90,377 | 10,356 records without a valid SMILES-to-XYZ atom mapping excluded (`isomorphism_failed`) |

The 18,164 excluded records are listed individually in
`data_indices/excluded_ids.csv` (`csv_idx, csd_code, reason`); the 90,377
retained records are listed in `data_indices/retained_ids.csv`
(`csv_idx, csd_code`). The two files partition the 108,541 source rows.

## Training stack vs inference stack

The training-stack parse uses RDKit 2024.03.6 and yields 100,733
transition-metal-containing records with parseable SMILES. The inference-stack
screen uses RDKit 2026.03.3 and yields 100,715. The 18-record difference is
caused by stricter main-group atom-valence sanitization in the newer RDKit
revision. The records and per-stack messages are in
`repro/round8_recon_20261016/id12b_diff.json`. None of the 18 belongs to the
13,556-complex test set.

## Reproduction entry points

| entry point | role |
|---|---|
| `analysis/rebuild_match_iso.py` | element-preserving graph matching (VF2++) and `sm2xyz` construction |
| `baselines/server_package/server_train_variants.py` | builds `ensemble_split_new.npz`, the 90,377-complex partitions used for training and evaluation |
| `data_indices/retained_ids.csv` | 90,377 retained `csv_idx, csd_code` pairs |
| `data_indices/excluded_ids.csv` | 18,164 excluded rows with reason codes |
| `data_indices/index_reconciliation.json` | machine-readable counts and source checksums |
| `analysis/verify_metal_weight.py` | independent recomputation of the main-domain metal-weight table |
| `repro/round8_recon_20261016/id12b_diff.json` | the 18-record two-stack parse difference |

The 1.1 GB graph tensor `new_domain_iso.pt` (md5
`3035057aeb8638680d676283ace2194b`) is not committed to the repository.
`retained_ids.csv` plus `data_indices/atom_maps.npz` contain the row-level
identifiers and atom mappings needed to rebuild the indexing without it.
