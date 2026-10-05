import requests
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


url = "https://gpcrdb.org/services/structure/"

response = requests.get(url)
response.raise_for_status()

structures = response.json()

print(f"Retrieved {len(structures)} structures")

structures_df = pd.DataFrame(structures)

print(f"Dataset shape: {structures_df.shape}")
print(structures_df.columns.tolist())

# Select experimentally determined human aminergic GPCR structure and with atleast one ligand

aminergic_structure = structures_df[ (structures_df["class"] == "Class A (Rhodopsin)") &
    (structures_df["family"].str.startswith("001_001")) &
    (structures_df["species"] == "Homo sapiens") & (structures_df["ligands"].apply(lambda x:len(x)>0))
].copy()

print(f"Human aminergic structures: {len(aminergic_structure)}")

# Ligand-GPCR metadata expansion

complexes_df = aminergic_structure.explode("ligands").reset_index(drop=True)

ligand_metadata = pd.json_normalize(complexes_df["ligands"])

complexes_df = pd.concat([complexes_df.drop(columns=["ligands"]).reset_index(drop=True), ligand_metadata.reset_index(drop=True)],axis=1)

complexes_df = complexes_df.rename(
    columns={
        "name": "ligand_name",
        "type": "ligand_type",
        "function": "ligand_function",
        "PDB" : "ligand_pdb_code",
        "SMILES": "ligand_smiles"
    }
)



complexes_df = complexes_df[
    [
        "pdb_code",
        "protein",
        "class",
        "family",
        "species",
        "resolution",
        "state",
        "ligand_name",
        "ligand_type",
        "ligand_function",
        "ligand_pdb_code",
        "ligand_smiles"
    ]
].copy()

complexes_df =  (
    complexes_df[
        ["pdb_code", "ligand_name", "ligand_function", "ligand_smiles"]
    ]
    .drop_duplicates(subset=["pdb_code", "ligand_name"])
)

print(f"GPCR–ligand complexes: {len(complexes_df)}")
complexes_df.head()

interactions_df = pd.read_csv("gpcr_interactions.csv")

print(f"Loaded interaction records: {len(interactions_df)}")

# def get_interactions(pdb_code):
#     url = f"https://gpcrdb.org/services/structure/{pdb_code}/interaction/"
    
#     response = requests.get(url)
#     response.raise_for_status()
    
#     return response.json()

# pdb_codes = complexes_df["pdb_code"].unique()

# print(f"Unique structures requiring interaction data: {len(pdb_codes)}")

# interaction_records = []

# for i,pdb_code in enumerate(pdb_codes, start=1):
#     print(f"[{i}/{len(pdb_codes)}] Fetching {pdb_code}...", flush=True)
#     interactions= get_interactions(pdb_code)

#     if interactions:
#         interaction_records.extend(interactions)

# print(f"Total interaction records: {len(interaction_records)}")


# interactions_df = pd.DataFrame(interaction_records)

# print(interactions_df.shape)
# interactions_df.head()

# interactions_df.to_csv("gpcr_interactions.csv", index=False)
# print("Interaction data saved.")
# MAtch interaction records to the corresponding GPCR-ligand complexes

interaction_df = interactions_df.merge(complexes_df,
on = ["pdb_code","ligand_name"],
how ="inner",
validate="many_to_one")

print(f"Matched interaction records: {len(interaction_df)}")
print(f"GPCR–ligand complexes with interactions: "
      f"{interaction_df[['pdb_code', 'ligand_name']].drop_duplicates().shape[0]}")

# Create binary interaction fingerprints
#Each coloumn represents a GPCR generic residue position

interactions_df["position"] = interaction_df["display_generic_number"]

fingerprints = pd.crosstab(
    [
        interaction_df["pdb_code"],
        interaction_df["ligand_name"],
        interaction_df["ligand_function"]
    ],
    interaction_df["display_generic_number"]
)

fingerprints = (fingerprints > 0).astype(int).reset_index()

print(f"Fingerprint dataset: {fingerprints.shape}")

# Save processed dataset
fingerprints.to_csv("gpcr_interaction_fingerprints.csv", index=False)

# Show the pharmacological classes available for modelling
print("\nLigand function distribution:")
print(fingerprints["ligand_function"].value_counts())


# Consolidate ligand functions into biologically meaningful classes

function_map = {
    "Agonist" : "Agonist-like",
    "Agonist (partial)" : "Agonist-like",
    "Allosteric agonist" : "Agonist-like",
    "Antagonist" : "Antagonist",
    "Inverse agonist" : "Inverse agonist",
    "PAM" : "Allosteric modulator",
    "Ago-PAM":"Allosteric modulator",
    "NAM": "Allosteric modulator"
}

ml_dataset = fingerprints.copy()

ml_dataset["pharmacological_class"] = (ml_dataset["ligand_function"].map(function_map))


# Remove enteries without a defined modelling class
ml_dataset= ml_dataset.dropna(subset=["pharmacological_class"])

print("ML dataset:", ml_dataset.shape)
print("\nClass distribution:")
print(ml_dataset["pharmacological_class"].value_counts())

ml_dataset.to_csv("gpcr_ml_dataset.csv", index=False)
print("\nML dataset saved.")


# Add receptor identity for grouped cross-validation

receptor_map = (
    aminergic_structure[["pdb_code", "protein"]].drop_duplicates(subset=["pdb_code"])
)

ml_dataset = ml_dataset.merge(
    receptor_map, on= "pdb_code", how="left"
)

#Define features and labels
metadata_columns = [
    "pdb_code", "ligand_name", "ligand_function", "pharmacological_class", "protein"
]

feature_coloumn = [
    column for column in ml_dataset.columns
    if column not in metadata_columns
]

X = ml_dataset[feature_coloumn].astype(float)
y = ml_dataset["pharmacological_class"]
groups = ml_dataset["protein"]

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Receptor groups: {groups.nunique()}")

# Grouped, stratified cross-validation
cv= StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    groups=groups,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1":"f1_macro"
    },
    n_jobs=-1
)

print("\nCross-validation results:")
print(
    f"Balanced accuracy: "
    f"{scores['test_balanced_accuracy'].mean():.3f} "
    f"+/- {scores['test_balanced_accuracy'].std():.3f}"
)

print(
    f"Macro F1: "
    f"{scores['test_macro_f1'].mean():.3f} "
    f"+/- {scores['test_macro_f1'].std():.3f}"
)

# Interpret the learned interaction fingerprints

model.fit(X,y)

feature_names = X.columns.tolist()

coeffecient_table = pd.DataFrame(
    model.coef_, index=model.classes_, columns=feature_names
)

for class_name in model.classes_:
    print(f"\nTop interaction positions for: {class_name}")

    top_features = (
        coeffecient_table.loc[class_name].sort_values(ascending=False).head(10)
    )

    for position, coefficient in top_features.items():
        print(f" {position}:{coefficient:.3f}")

# Contact frequency of each residue position within each pharmacological class

feature_data = ml_dataset[feature_coloumn].copy()
feature_data["pharmacological_class"] = ml_dataset["pharmacological_class"].values

contact_frequency = (
    feature_data.groupby("pharmacological_class")[feature_coloumn].mean()
)

# Show the most variable interaction position across classes

position_variability = (
    contact_frequency.max(axis=0) -contact_frequency.min(axis=0)
).sort_values(ascending=False)


print("Most class-discriminating interaction positions:\n")

for position in position_variability.head(20).index:
    print(f"{position}:")
    print(contact_frequency[position].sort_values(ascending=False).round(3).to_dict())
    print()


# Define binding mode independently from pharmacological class

allosteric_functions = [
    "PAM",
    "Ago-PAM",
    "NAM",
    "Allosteric agonist"
]

ml_dataset["binding_mode"] = np.where(
    ml_dataset["ligand_function"].isin(allosteric_functions),
    "Allosteric",
    "Orthosteric"
)

print(ml_dataset["binding_mode"].value_counts())
# Orthosteric vs allosteric classification

X = ml_dataset[feature_coloumn].astype(float)
y = ml_dataset["binding_mode"]
groups = ml_dataset["protein"]

binary_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

binary_scores = cross_validate(
    binary_model,
    X,
    y,
    cv=cv,
    groups=groups,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro"
    },
    n_jobs=-1
)

print(
    f"Orthosteric vs allosteric balanced accuracy: "
    f"{binary_scores['test_balanced_accuracy'].mean():.3f} "
    f"+/- {binary_scores['test_balanced_accuracy'].std():.3f}"
)

print(
    f"Orthosteric vs allosteric macro F1: "
    f"{binary_scores['test_macro_f1'].mean():.3f} "
    f"+/- {binary_scores['test_macro_f1'].std():.3f}"
)

# Pharmacological classification among orthosteric ligands
orthosteric = ml_dataset[
    (ml_dataset["binding_mode"] == "Orthosteric") &
    (ml_dataset["pharmacological_class"].isin(
        ["Agonist-like", "Antagonist", "Inverse agonist"]
    ))
].copy()

print("Orthosteric dataset:", orthosteric.shape)
print("\nClass distribution:")
print(orthosteric["pharmacological_class"].value_counts())


X_ortho = orthosteric[feature_coloumn].astype(float)
y_ortho = orthosteric["pharmacological_class"]
groups_ortho = orthosteric["protein"]


ortho_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

ortho_scores = cross_validate(
    ortho_model,
    X_ortho,
    y_ortho,
    cv=cv,
    groups=groups_ortho,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro"
    },
    n_jobs=-1
)

print("\nOrthosteric pharmacology classification:")
print(
    f"Balanced accuracy: "
    f"{ortho_scores['test_balanced_accuracy'].mean():.3f} "
    f"+/- {ortho_scores['test_balanced_accuracy'].std():.3f}"
)

print(
    f"Macro F1: "
    f"{ortho_scores['test_macro_f1'].mean():.3f} "
    f"+/- {ortho_scores['test_macro_f1'].std():.3f}"
)

# Cross-validated predictions for orthosteric ligands
ortho_predictions = cross_val_predict(
    ortho_model,
    X_ortho,
    y_ortho,
    cv=cv,
    groups=groups_ortho,
    n_jobs=-1
)

print("Classification report:\n")
print(
    classification_report(
        y_ortho,
        ortho_predictions,
        digits=3
    )
)

print("Confusion matrix:")
print(confusion_matrix(
    y_ortho,
    ortho_predictions,
    labels=["Agonist-like", "Antagonist", "Inverse agonist"]
))

# Plot 1: cross-validated confusion matrix

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_ortho,
    ortho_predictions,
    labels=["Agonist-like", "Antagonist", "Inverse agonist"],
    normalize="true",
    values_format=".2f",
    ax=ax
)

ax.set_title("Cross-validated prediction of orthosteric ligand pharmacology")
plt.tight_layout()
plt.savefig("Confusion_matrix.png", format = "png",dpi=300,bbox_inches = "tight" )
plt.show()

# Plot 2: Interaction frequency across pharmacological classes

top_positions = position_variability.head(15).index

heatmap_data = contact_frequency.loc[
    ["Agonist-like", "Antagonist", "Inverse agonist", "Allosteric modulator"],
    top_positions
]

fig, ax = plt.subplots(figsize=(14, 5))

im = ax.imshow(
    heatmap_data.values,
    aspect="auto",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(len(top_positions)))
ax.set_xticklabels(
    top_positions,
    rotation=60,
    ha="right"
)

ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

ax.set_xlabel("GPCR generic residue position")
ax.set_ylabel("Pharmacological class")
ax.set_title("Interaction frequency at class-discriminating GPCR positions")

fig.colorbar(
    im,
    ax=ax,
    label="Fraction of complexes with interaction"
)

plt.tight_layout()

plt.savefig(
    "gpcr_interaction_frequency_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Plot 1: Pharmacological class distribution

class_counts = ml_dataset["pharmacological_class"].value_counts()

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    class_counts.index,
    class_counts.values
)

ax.set_xlabel("Pharmacological class")
ax.set_ylabel("Number of GPCR–ligand complexes")
ax.set_title("GPCR–ligand dataset composition")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "pharmacological_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()