"""Contrôle du fichier submission.csv avant envoi à Kaggle (Rossmann).

Ne vérifie pas le score (inconnu), mais tout ce qui peut faire rejeter le fichier
et les incohérences grossières de prédiction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SUB = ROOT / "submission.csv"
TEST = ROOT / "test.csv"
SAMPLE = ROOT / "sample_submission.csv"
TRAIN = ROOT / "train.csv"

erreurs = []
alertes = []


def check(condition, message):
    if not condition:
        erreurs.append(message)


# ---------------------------------------------------------------- lecture
sub = pd.read_csv(SUB)
test = pd.read_csv(TEST, parse_dates=["Date"])

print("=== 1. Structure du fichier ===")
print("Colonnes :", list(sub.columns))
print("Lignes   :", len(sub))
print("Lignes test.csv :", len(test))

check(list(sub.columns) == ["Id", "Sales"],
      f"Colonnes attendues ['Id', 'Sales'], trouvé {list(sub.columns)}")
check(len(sub) == len(test),
      f"{len(sub)} lignes au lieu de {len(test)}")

# ------------------------------------------------------------------ Id
print("\n=== 2. Colonne Id ===")
print("Type :", sub["Id"].dtype)
print("Doublons :", int(sub["Id"].duplicated().sum()))
print("Min / Max :", sub["Id"].min(), "/", sub["Id"].max())

check(sub["Id"].notna().all(), "Id contient des valeurs manquantes")
check(not sub["Id"].duplicated().any(), "Id contient des doublons")
check(set(sub["Id"]) == set(test["Id"]),
      "Les Id ne correspondent pas exactement à ceux de test.csv")
check((sub["Id"].sort_values().to_numpy() ==
       np.sort(test["Id"].to_numpy())).all(),
      "Les Id ne sont pas identiques (ordre ou valeurs) à test.csv")

# Fichier de référence Kaggle, s'il est présent dans le dossier
if SAMPLE.exists():
    sample = pd.read_csv(SAMPLE)
    print("sample_submission.csv trouvé :", len(sample), "lignes")
    check(list(sample.columns) == list(sub.columns),
          "Colonnes différentes de sample_submission.csv")
    check(len(sample) == len(sub), "Nb de lignes différent de sample_submission.csv")
    check(sample["Id"].tolist() == sub["Id"].tolist(),
          "Ordre des Id différent de sample_submission.csv")
else:
    alertes.append("sample_submission.csv absent : comparaison d'ordre non faite "
                   "(télécharge-le depuis Kaggle pour un contrôle complet).")

# --------------------------------------------------------------- Sales
print("\n=== 3. Colonne Sales ===")
s = sub["Sales"]
print("Manquants      :", int(s.isna().sum()))
print("Valeurs infinies:", int(np.isinf(s.dropna()).sum()))
print("Négatives      :", int((s < 0).sum()))
print("Zéros          :", int((s == 0).sum()))
print(s.describe().to_string())

check(s.notna().all(), "Sales contient des NaN (Kaggle rejette le fichier)")
check(np.isfinite(s).all(), "Sales contient des valeurs infinies")
check((s >= 0).all(), "Sales contient des valeurs négatives")

# ------------------------------------------- cohérence jours fermés/ouverts
print("\n=== 4. Cohérence avec Open de test.csv ===")
ctrl = test[["Id", "Store", "Date", "Open"]].merge(
    sub, on="Id", how="left", validate="one_to_one"
)
ctrl["Open"] = ctrl["Open"].fillna(1)  # hypothèse documentée : Store 622 -> ouvert

fermes = ctrl["Open"] == 0
ouverts = ctrl["Open"] == 1

print("Lignes fermées :", int(fermes.sum()),
      "| Sales non nulle parmi elles :",
      int((ctrl.loc[fermes, "Sales"] != 0).sum()))
print("Lignes ouvertes:", int(ouverts.sum()),
      "| Sales nulle ou négative parmi elles :",
      int((ctrl.loc[ouverts, "Sales"] <= 0).sum()))

check((ctrl.loc[fermes, "Sales"] == 0).all(),
      "Certains jours fermés (Open=0) ont Sales != 0")
check((ctrl.loc[ouverts, "Sales"] > 0).all(),
      "Certains jours ouverts ont Sales <= 0")

# --------------------------------------------- plausibilité des prédictions
print("\n=== 5. Plausibilité des prédictions ===")
print("Sales moyenne prédite (jours ouverts) :",
      round(ctrl.loc[ouverts, "Sales"].mean(), 2))

# Référence : ventes réelles des jours ouverts du train
ouverts_train = pd.read_csv(
    TRAIN, usecols=["Store", "Sales", "Open"], nrows=None
)
ouverts_train = ouverts_train.loc[
    (ouverts_train["Open"] == 1) & (ouverts_train["Sales"] > 0)
]
print("Sales moyenne réelle train (jours ouverts) :",
      round(ouverts_train["Sales"].mean(), 2))
print("Sales médiane réelle train  (jours ouverts) :",
      round(ouverts_train["Sales"].median(), 2))

moyenne_train = ouverts_train.groupby("Store")["Sales"].mean()
moyenne_pred = ctrl.loc[ouverts].groupby("Store")["Sales"].mean()

ratio = (moyenne_pred / moyenne_train).dropna()
print("\nRatio prévu/réel par magasin :")
print(ratio.describe().to_string())

hors = ratio[(ratio < 0.5) | (ratio > 2.0)]
print("\nMagasins hors de [0.5 ; 2.0] :", len(hors))
if len(hors) > 0:
    print(hors.sort_values().to_string())
    alertes.append(f"{len(hors)} magasin(s) avec des prédictions très éloignées "
                   "de leur historique : à inspecter, pas forcément une erreur.")

# ------------------------------------------------------------------ verdict
print("\n=== Verdict ===")
if erreurs:
    print("ERREURS BLOQUANTES :")
    for e in erreurs:
        print(" -", e)
else:
    print("Format valide : le fichier peut être envoyé à Kaggle.")

if alertes:
    print("\nAlertes (non bloquantes) :")
    for a in alertes:
        print(" -", a)
