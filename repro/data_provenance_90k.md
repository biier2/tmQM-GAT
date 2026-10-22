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
| `analysis/rebuild_match_iso.py` | element-preserving graph matching (VF2++) and `sm2xyz` construction; writes `new_domain_iso.pt` with the paper split (`te_idx = RandomState(42).permutation(90377)[:13556]`) |
| `baselines/server_package/server_train_variants.py` | builds `ensemble_split_new.npz`, the 90,377-complex partitions used for training and evaluation |
| `data_indices/retained_ids.csv` | 90,377 retained `csv_idx, csd_code` pairs |
| `data_indices/excluded_ids.csv` | 18,164 excluded rows with reason codes |
| `data_indices/index_reconciliation.json` | machine-readable counts and source checksums |
| `analysis/verify_metal_weight.py` | independent recomputation of the main-domain metal-weight table |
| `repro/verify_domain_content.py` | content-level verification of a rebuilt `new_domain_iso.pt` against `repro/domain_content_digests.json` |
| `repro/round8_recon_20261016/id12b_diff.json` | the 18-record two-stack parse difference |

## Rebuild verification

The 1.1 GB graph tensor `new_domain_iso.pt` is not committed to the repository.
The authors' copy is 1,107,170,889 bytes with file md5
`3035057aeb8638680d676283ace2194b`, but **`torch.save` is not
byte-deterministic**: saving the same dictionary twice gives different file
bytes, so a rebuilt tensor will not match that md5 even when its contents are
identical. Use the content digests below instead.

```bash
python analysis/rebuild_match_iso.py                  # -> new_domain_iso.pt
python repro/verify_domain_content.py --data new_domain_iso.pt
```

The verifier hashes each array as contiguous little-endian C-order bytes in
domain order, concatenates list elements in order for the per-complex feature
lists, joins string lists with newlines (UTF-8), and checks the domain size,
the seed-42 test split and `data_indices/retained_ids.csv`.

| field | bytes | content md5 |
|---|---:|---|
| `csv_idx` | 723,016 | `ac8ba24532767b542dfb7acda4a746ec` |
| `y_all` | 2,892,064 | `ea285df3e7941743f4e0d14d3ae6f1dc` |
| `te_idx` | 54,224 | `9f1dda76e12436fe29834023ba6cb0a4` |
| `y_mean` | 32 | `1167e220507f7a16a9c80e4842e9ece3` |
| `y_std` | 32 | `edd852208712a46d468f6e86c91f347a` |
| `csd_codes` (newline-joined) | 90,377 entries | `3c3e72a33ad53c20d6e2ef2f3d8a65d5` |
| `match_src` (newline-joined) | 90,377 entries | `285d264f84faf1e6a4a10b8f1bb8418a` |
| `atoms_filt` (list) | 47,175,864 | `b4d741ff54da47e7bb42a77d87e3e4e0` |
| `node8s` (list) | 100,655,008 | `943e5c5aca5bb474583357943a5de311` |
| `node3s` (list) | 37,745,628 | `e385d7203c14d1d4c5f1a1a88a2a98ef` |
| `ei_ls` (list) | 115,219,744 | `2128bb8997bffdf86155162f4e1e7e94` |
| `e6_ls` (list) | 172,829,616 | `c33dc5d5cea39e85fdf43e3e825b043c` |
| `rbf_ls` (list) | 460,878,976 | `f222d541774f1267537f8d4fd2f7b4e5` |
| `bo1_ls` (list) | 28,804,936 | `6fda7774eda04ade343e4d413e783fce` |

`match_src` counts are `BO0.2` 61,759, `dist0.45` 27,273, `dist0.60` 1,345.
The machine-readable copy of this table is `repro/domain_content_digests.json`.
`retained_ids.csv` plus `data_indices/atom_maps.npz` contain the row-level
identifiers and atom mappings needed to rebuild the indexing without the
tensor; the feature arrays additionally require the public XYZ/BO archives and
the pinned RDKit/NetworkX stack in `environment_train.yml`.
