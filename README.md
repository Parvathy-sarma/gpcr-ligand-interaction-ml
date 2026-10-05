# Learning GPCR Ligand Recognition from Protein–Ligand Interaction Fingerprints

## Introduction

G protein-coupled receptors (GPCRs) are membrane proteins that translate signals from molecules outside the cell into cellular responses. Their interactions with small molecules determine whether a receptor is activated, inhibited, or modulated, making GPCR–ligand recognition central to understanding pharmacology and drug design.

Recent advances in artificial intelligence (AI) and machine learning (ML) provide new ways to analyse large structural datasets and identify patterns that are difficult to recognise by manual inspection alone. This project is a small computational study exploring whether AI/ML can help decode patterns of GPCR–ligand interactions and their relationship to ligand pharmacology.

## Scientific Question

**Can protein–ligand interaction fingerprints reveal patterns that distinguish GPCR ligand binding modes and pharmacological classes?**

The rationale is that different ligands do not interact with a GPCR in exactly the same way. They may contact different residues within the receptor, and these interaction patterns may contain information about where a ligand binds and how it acts on the receptor.

Rather than trying to predict drug activity directly from molecular structure, this project focuses on the structural interaction layer first: **which GPCR residues are contacted by each ligand, and what can these contact patterns tell us?**

A second motivation is to connect structural bioinformatics with molecular simulation. If static interaction patterns can capture some aspects of ligand recognition but not receptor function completely, receptor conformational dynamics and interaction persistence from molecular dynamics (MD) simulations become a natural next step.

## Dataset

### Data source

The initial GPCRdb dataset contained 2,577 GPCR structure entries. I first selected 436 human aminergic Class A GPCR structures. Because some structures contain more than one ligand, expanding the ligand information produced 488 GPCR–ligand complex entries, where each entry represents one specific ligand associated with one GPCR structure.

Interaction data were then retrieved for these complexes. After excluding complexes without available interaction information and entries with an undefined pharmacological class, the final dataset contained 472 GPCR–ligand complexes.

Each complex was represented numerically as an interaction fingerprint consisting of 225 GPCR generic residue-position features. Each feature records whether an interaction was observed at that position (1) or not (0). The 472 complexes belonged to 35 distinct receptor groups, which were used for grouped cross-validation to reduce information leakage between structurally related receptor entries.

After quality filtering and removal of entries without a defined pharmacological class, the final ML dataset contained:

- **472 GPCR–ligand complexes**
- **225 GPCR interaction-position features**
- **35 receptor groups** used for grouped cross-validation

For the binding-mode analysis, the 472 complexes contained:

- **423 orthosteric complexes**
- **49 allosteric complexes**

### What types of ligands are present?

The original dataset contains several pharmacological annotations, including:

- Agonists
- Partial agonists
- Antagonists
- Inverse agonists
- Positive allosteric modulators (PAMs)
- Ago-PAMs
- Negative allosteric modulators (NAMs)
- Allosteric agonists
- A small number of entries with unknown function

For the ML analysis, these annotations were consolidated into four broader pharmacological-effect classes:

| Original annotations | ML class |
|---|---|
| Agonist, partial agonist, allosteric agonist | **Agonist-like** |
| Antagonist | **Antagonist** |
| Inverse agonist | **Inverse agonist** |
| PAM, Ago-PAM, NAM | **Allosteric modulator** |

The final class distribution was 347 agonist-like, 45 allosteric modulator, 41 inverse agonist, and 39 antagonist complexes.

### Orthosteric vs allosteric ligands

Orthosteric ligands bind to the receptor's main binding site, which is generally the site used by the receptor's endogenous ligand. Allosteric ligands bind at a different site on the receptor and can modify receptor activity from that alternative location.

It is important to distinguish **pharmacological effect** from **binding mode**. For example, an allosteric agonist can produce an agonist-like effect while binding at an allosteric site. Therefore, the project keeps these two concepts separate: pharmacological annotations are used to define the four ML effect classes above, while binding mode is assigned independently from the original ligand annotations.

For the binding-mode analysis:

- **Orthosteric:** agonists, partial agonists, antagonists, and inverse agonists
- **Allosteric:** PAMs, Ago-PAMs, NAMs, and allosteric agonists

### How are the interactions represented numerically?

Each ligand–GPCR complex is represented as an **interaction fingerprint**.

GPCRdb provides generic residue positions that allow equivalent receptor positions to be compared across different GPCRs. Examples include positions such as `3.32`, `5.42`, and `6.52`. Interaction features such as `3.32x32` or `7.45` represent standardized GPCR structural positions, where the first number denotes the transmembrane helix and the following number identifies the equivalent residue position within that helix, allowing ligand interactions to be compared consistently across different GPCRs.

For each GPCR–ligand complex:

- **1** means that an interaction with that generic receptor position is observed in the structural data.
- **0** means that no interaction with that position is observed.

This converts a structural interaction pattern into a binary numerical representation that can be analysed using machine-learning methods.

Therefore, the model's input is not simply the ligand or receptor name. It is a numerical representation of **which receptor positions are involved in the ligand interaction**.

## Computational Workflow

The analysis was carried out in Python using GPCRdb data and standard scientific and machine-learning libraries.

The workflow was:

1. Retrieve GPCR structure metadata from GPCRdb.
2. Select human aminergic Class A GPCR structures.
3. Expand structures containing multiple ligands into individual GPCR–ligand complexes.
4. Retrieve protein–ligand interaction data for each structure.
5. Match interaction records to their corresponding GPCR–ligand complexes.
6. Convert residue contacts into binary interaction fingerprints using GPCR generic numbering.
7. Group ligand annotations into broader pharmacological-effect classes.
8. Train an interpretable logistic-regression classifier.
9. Use **StratifiedGroupKFold** so structures from the same receptor are kept within the same cross-validation group.
10. Examine which interaction positions contribute to classification and visualise the results.

## Results and Interpretation

### 1. Dataset composition

![GPCR–ligand dataset composition](figures/pharmacological_class_distribution.png)

The dataset is dominated by agonist-like complexes, while antagonist, inverse-agonist, and allosteric-modulator classes are smaller. This class imbalance is important to consider when interpreting the ML results, which is why balanced accuracy and macro F1 were used rather than relying on accuracy alone.

### 2. Which GPCR positions distinguish the classes?

![Interaction frequency heatmap](figures/gpcr_interaction_frequency_heatmap.png)

The heatmap shows how often different GPCR generic positions are contacted by each pharmacological class.

A clear pattern is visible: several receptor positions are contacted very frequently by orthosteric ligands but rarely by allosteric modulators. This suggests that the interaction fingerprint contains strong information about **where a ligand binds**.

The differences between agonist-like, antagonist, and inverse-agonist ligands are much smaller. Many of the same residues are contacted across these classes, indicating that simply knowing whether a residue is contacted may not be enough to determine the functional effect of an orthosteric ligand.

### 3. Can interaction fingerprints distinguish binding mode?

![Orthosteric pharmacology confusion matrix](figures/orthosteric_pharmacology_confusion_matrix.png)

Interaction fingerprints distinguished orthosteric from allosteric ligands very strongly:

- **Balanced accuracy: 0.988 ± 0.025**
- **Macro F1: 0.988 ± 0.025**

This indicates that the structural contact pattern contains strong information about **where a ligand binds** on the receptor.

### 4. Can interaction fingerprints distinguish orthosteric pharmacology?

For the orthosteric subset, the model was asked to distinguish agonist-like, antagonist, and inverse-agonist pharmacology.

- **423 orthosteric GPCR–ligand complexes**
- **343 agonist-like**
- **39 antagonist**
- **41 inverse agonist**

Performance was substantially lower:

- **Balanced accuracy: 0.535 ± 0.103**
- **Macro F1: 0.443 ± 0.093**

The confusion matrix shows that agonist-like ligands are recognised relatively well, whereas antagonist and inverse-agonist ligands are frequently confused with each other.

### What does this mean?

The results suggest that **static interaction fingerprints are very informative about binding mode, but less informative about functional receptor response**.

In simple terms, the model can learn something close to:

> **“Where is this ligand interacting with the receptor?”**

much more easily than:

> **“What functional effect will this ligand produce?”**

This distinction is important for GPCR drug discovery. Agonism, antagonism, and inverse agonism depend not only on which residues are contacted, but also on receptor conformational state and how ligand interactions influence receptor dynamics.

## Conclusion

This project provides a small computational demonstration of how structural GPCR data can be combined with machine learning to study ligand recognition.

The main finding is that **GPCR–ligand interaction fingerprints strongly distinguish orthosteric and allosteric binding modes, but static contacts alone provide only limited information for separating agonist, antagonist, and inverse-agonist function among orthosteric ligands.**

The result points naturally toward a next step: combining structural interaction fingerprints with **molecular dynamics simulations and interaction persistence** to capture how GPCR–ligand interactions evolve over time and how they relate to receptor activation states.

This project therefore represents a first step toward integrating **structural bioinformatics, machine learning, and molecular simulation** for GPCR-focused structure-based drug discovery.

## Tools

Python · pandas · NumPy · Matplotlib · scikit-learn · GPCRdb

## Project Status

This is a **pilot computational study** intended to explore the relationship between GPCR interaction fingerprints and ligand pharmacology. The results should be interpreted as dataset-specific structural patterns rather than as a general-purpose pharmacological predictor.
