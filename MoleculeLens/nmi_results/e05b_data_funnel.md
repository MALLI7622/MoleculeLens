ChEMBL release: ChEMBL_37

| Step | Rows left | Removed |
|---|---|---|
| ChEMBL drug_mechanism table, all rows | 7,561 | - |
| molecular_mechanism = 1 (the e05 download filter) | 7,559 | 2 |
| has a molecule ChEMBL id | 7,559 | 0 |
| molecule has a SMILES structure | 5,411 | 2,148 |
| mechanism text present | 5,411 | 0 |
| mechanism text longer than 15 characters | 5,066 | 345 |
| unique (molecule, target) pair | 5,035 | 31 |

**Rows removed for lacking a SMILES, by molecule type** (rows / distinct molecules):

| Molecule type | Rows | Molecules |
|---|---|---|
| Unknown | 677 | 586 |
| Antibody | 632 | 567 |
| Protein | 295 | 237 |
| Small molecule | 130 | 122 |
| Antibody drug conjugate | 123 | 63 |
| Enzyme | 85 | 73 |
| Vaccine component | 77 | 69 |
| Oligonucleotide | 75 | 72 |
| Gene | 29 | 28 |
| Oligosaccharide | 14 | 14 |
| Cell | 11 | 11 |

Molecule types kept: {'Small molecule': 3970, 'Protein': 122, 'Unknown': 9, 'Oligosaccharide': 5, 'Oligonucleotide': 4}
Final: 5,035 pairs, 3,816 molecules; by max phase: {'4': 2769, '2': 1292, '3': 680, '1': 268, '-1': 26}
Check vs committed CSV: {'committed_rows': 5035, 'rerun_rows': 5035, 'pairs_in_both': 5035, 'only_in_rerun': 0, 'only_in_committed': 0}
