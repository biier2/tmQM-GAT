# Fig. 5 recomputation — consolidated tables

test-set run: n=2000, test fraction 100.0%
full-domain run: n=1805, test fraction 16.4%

## T1. Published donor definition (A_old = COORD+REMOTE_H), full-domain vs test-set pool

| property | published | reproduced (full domain) | **test set** | repro - published | test - full | test - published |
|---|---|---|---|---|---|---|
| HL-Gap | 31.3% | 30.9% | **31.9%** | -0.39 | +0.94 | +0.56 |
| HOMO | 47.6% | 47.6% | **47.2%** | -0.05 | -0.33 | -0.38 |
| LUMO | 44.1% | 44.2% | **44.0%** | +0.09 | -0.21 | -0.12 |
| Dipole | 26.1% | 25.8% | **25.5%** | -0.27 | -0.31 | -0.59 |
| Metal q | 30.5% | 30.4% | **30.5%** | -0.08 | +0.07 | -0.01 |
| E_elec | 14.0% | 13.8% | **13.8%** | -0.18 | -0.01 | -0.19 |
| E_disp | 23.5% | 23.4% | **23.4%** | -0.10 | -0.04 | -0.14 |
| Polarizability | 21.5% | 21.5% | **21.4%** | +0.01 | -0.08 | -0.07 |

reproduction fidelity: mean |repro - published| = 0.15 pp; max = 0.39 pp
test set vs full domain: mean |test - full| = 0.25 pp; max = 0.94 pp

## T2. Six-way atom typing, test set (share of total |attribution|, %)

| property | METAL | MBC | **COORD** | REMOTE_H | OTHER | REMOTE_C | A_old | **A_new** | B_old | **B_new** |
|---|---|---|---|---|---|---|---|---|---|---|
| HL-Gap | 9.2 | 14.7 | **23.0** | 13.5 | 7.9 | 49.9 | 31.9 | **23.0** | 39.6 | 29.5 |
| HOMO | 15.2 | 27.3 | **38.4** | 14.6 | 9.3 | 21.7 | 47.2 | **38.4** | 61.6 | 50.7 |
| LUMO | 15.3 | 26.9 | **34.9** | 14.6 | 8.6 | 25.1 | 44.0 | **34.9** | 58.2 | 47.3 |
| Dipole | 10.4 | 12.2 | **17.1** | 12.4 | 6.3 | 56.7 | 25.5 | **17.1** | 31.9 | 22.6 |
| Metal q | 21.7 | 12.8 | **22.1** | 12.7 | 6.0 | 40.2 | 30.5 | **22.1** | 37.2 | 27.7 |
| E_elec | 30.5 | 9.2 | **8.4** | 7.8 | 4.4 | 50.2 | 13.8 | **8.4** | 18.7 | 12.8 |
| E_disp | 39.2 | 20.2 | **19.2** | 7.0 | 6.1 | 25.7 | 23.4 | **19.2** | 34.1 | 28.8 |
| Polarizability | 42.5 | 18.0 | **17.3** | 6.8 | 5.7 | 25.6 | 21.4 | **17.3** | 31.0 | 25.9 |

## T3. Per-atom intensity ratio, atom-pooled (1.0 = the average atom)

| property | METAL | MBC | COORD | REMOTE_H | REMOTE_C |
|---|---|---|---|---|---|
| HL-Gap | 3.08 | 1.13 | **2.28** | 1.18 | 0.72 |
| HOMO | 4.41 | 2.82 | **3.45** | 1.81 | 0.30 |
| LUMO | 4.53 | 2.74 | **3.24** | 1.73 | 0.33 |
| Dipole | 2.99 | 0.77 | **1.63** | 1.15 | 0.85 |
| Metal q | 7.94 | 1.39 | **1.89** | 1.37 | 0.54 |
| E_elec | 9.14 | 0.68 | **0.80** | 0.75 | 0.77 |
| E_disp | 12.45 | 1.43 | **1.82** | 0.67 | 0.44 |
| Polarizability | 13.35 | 1.32 | **1.67** | 0.63 | 0.44 |

## T4. Size-matched exchangeability null (per-molecule paired z)

| property | COORD mean | **COORD median** | REMOTE_H mean | **REMOTE_H median** | MBC mean | METAL mean | REMOTE_C mean |
|---|---|---|---|---|---|---|---|
| HL-Gap | 2.11 | **+2.27** | 0.40 | **-0.04** | 0.59 | 1.95 | -2.58 |
| HOMO | 2.51 | **+2.71** | 0.38 | **-0.27** | 1.72 | 1.94 | -3.25 |
| LUMO | 2.26 | **+2.33** | 0.39 | **-0.25** | 1.71 | 1.92 | -3.06 |
| Dipole | 1.22 | **+1.17** | 0.32 | **+0.05** | -0.03 | 1.96 | -1.57 |
| Metal q | 1.55 | **+1.39** | 0.22 | **+0.05** | 0.14 | 3.95 | -2.69 |
| E_elec | -0.33 | **-0.40** | -0.38 | **-0.38** | -0.23 | 5.47 | -1.57 |
| E_disp | 0.88 | **+0.52** | -0.33 | **-0.33** | 0.79 | 5.24 | -2.79 |
| Polarizability | 0.67 | **+0.34** | -0.32 | **-0.33** | 0.55 | 5.38 | -2.60 |

n = 1879 complexes. z is the paired z against the exchangeability null E[share] = n_k/N.

The mean and the median disagree for uncoordinated donor-element atoms: their distribution is right-skewed, so the median is the honest summary of a *typical* complex. For HOMO the median is -0.2 (a typical complex gives uncoordinated donors no more weight than a random set of the same size) while the mean is +0.4, i.e. a minority of complexes carry a large remote contribution. Coordinated donors are unambiguous: median +2.7.

## T4b. Distribution-free form of T4 (test set)

Fraction of complexes in which the class exceeds its own size-matched expectation, and the mean excess in percentage points.

| property | COORD above | REMOTE_H above | MBC above | METAL above | REMOTE_C above | COORD excess | REMOTE_H excess | MBC excess |
|---|---|---|---|---|---|---|---|---|
| HL-Gap | **84%** | 49% | 58% | 82% | 8% | +11.3 pp | +2.1 pp | +1.3 pp |
| HOMO | **85%** | 36% | 74% | 84% | 0% | +26.7 pp | +3.2 pp | +13.8 pp |
| LUMO | **81%** | 38% | 72% | 84% | 0% | +23.3 pp | +3.2 pp | +13.4 pp |
| Dipole | **75%** | 51% | 48% | 77% | 19% | +5.5 pp | +1.0 pp | -1.3 pp |
| Metal q | **84%** | 52% | 49% | 94% | 2% | +10.5 pp | +1.3 pp | -0.7 pp |
| E_elec | **21%** | 14% | 31% | 100% | 2% | -3.2 pp | -3.6 pp | -4.3 pp |
| E_disp | **81%** | 13% | 84% | 100% | 0% | +7.5 pp | -4.3 pp | +6.7 pp |
| Polarizability | **74%** | 12% | 79% | 100% | 0% | +5.6 pp | -4.5 pp | +4.5 pp |

Under the null a class is above its expectation in 50% of complexes. COORD is at 75-85% for the five electronic/electrostatic properties; REMOTE_H is at 36-52%, i.e. at chance for a typical complex, while its mean paired z stays positive (T4) because a minority of complexes carry a strong remote contribution. Both statements are true and must be reported together: panel (a) of the published figure is dominated by coordinated donors, but not exclusively so.

## T5. Is the Fig. 5a -> 5b increase explained by construction?

Observed increase vs the increase an exchangeable (size-matched) null predicts, which is exactly the growth of the set.

| property | A_old | B_old | observed increase | set grew by | ratio |
|---|---|---|---|---|---|
| HL-Gap | 31.9% | 39.6% | +7.7 pp | +7.2 pp | 1.07x |
| HOMO | 47.2% | 61.6% | +14.4 pp | +7.2 pp | 1.99x |
| LUMO | 44.0% | 58.2% | +14.2 pp | +7.2 pp | 1.97x |
| Dipole | 25.5% | 31.9% | +6.4 pp | +7.2 pp | 0.89x |
| Metal q | 30.5% | 37.2% | +6.7 pp | +7.2 pp | 0.93x |
| E_elec | 13.8% | 18.7% | +4.8 pp | +7.2 pp | 0.67x |
| E_disp | 23.4% | 34.1% | +10.7 pp | +7.2 pp | 1.49x |
| Polarizability | 21.4% | 31.0% | +9.5 pp | +7.2 pp | 1.32x |

## T6. Atom budget (test-set sample)

| class | atoms | share of all atoms |
|---|---|---|
| METAL | 2,000 | 2.88% |
| MBC | 4,108 | 5.91% |
| COORD | 6,789 | 9.76% |
| REMOTE_H | 5,427 | 7.80% |
| OTHER | 1,128 | 1.62% |
| REMOTE_C | 50,098 | 72.03% |

total atoms: 69,550; mean heavy atoms per complex: 34.8

Atom-pooled: A_old (published donor set) = 12,216 atoms = 17.56% of all atoms, of which REMOTE_H = 5,427 = 7.80 pp **(44.4%) are NOT bonded to the metal**.
Per-complex mean (the quantity that enters Fig. 5): A_old = 19.47%, REMOTE_H = 8.53% (43.8%).
