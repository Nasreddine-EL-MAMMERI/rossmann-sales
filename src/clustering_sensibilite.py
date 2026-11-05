"""Analyse de sensibilité de la segmentation.

Question traitée : la redondance entre coef_variation (0.82), uplift_promo
et ratio_samedi_hp (-0.68) sur-pondère l'axe "dépendance promotionnelle".
Le découpage survit-il si on retire une de ces variables ?

Méthode : on réexécute exactement le même pipeline (standardisation, KMeans
k=3, graine fixée) en retirant ou ajoutant une variable à la fois, puis on
mesure l'accord avec la segmentation de référence par l'indice ARI.

ARI > 0.80 : conclusion inchangée
ARI 0.60-0.80 : structure proche mais des magasins changent de segment
ARI < 0.60 : la variable retirée portait réellement la séparation
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

K = 3
SEED = 42

# ================================================================
# 1. Variables (mêmes définitions que clustering_final.py)
# ================================================================
profil = pd.read_csv(ROOT / "profil_magasin_prepare.csv")

train = pd.read_csv(
    ROOT / "train.csv",
    usecols=["Store", "DayOfWeek", "Sales", "Promo", "Open"],
)
ouvert = train.loc[(train["Open"] == 1) & (train["Sales"] > 0)]
hors_promo = ouvert.loc[ouvert["Promo"] == 0]

uplift = (
    ouvert.loc[ouvert["Promo"] == 1].groupby("Store")["Sales"].mean()
    / hors_promo.groupby("Store")["Sales"].mean()
).rename("uplift_promo")

ratio = (
    hors_promo.loc[hors_promo["DayOfWeek"] == 6].groupby("Store")["Sales"].mean()
    / hors_promo.loc[hors_promo["DayOfWeek"].isin([1, 2, 3, 4, 5])]
      .groupby("Store")["Sales"].mean()
).rename("ratio_samedi_hp")

profil = profil.merge(
    pd.concat([uplift, ratio], axis=1).reset_index(),
    on="Store", how="left", validate="one_to_one",
)
for col in ["uplift_promo", "ratio_samedi_hp"]:
    profil[col] = profil[col].fillna(profil[col].median())


def segmenter(features):
    X = StandardScaler().fit_transform(profil[features])
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    return labels, silhouette_score(X, labels)


BASE = [
    "log_ca_median", "coef_variation", "log_competition",
    "ouvert_dimanche", "uplift_promo", "ratio_samedi_hp",
]

labels_ref, sil_ref = segmenter(BASE)

variantes = {
    "Référence (6 variables)": BASE,
    "Sans log_competition\n(eta2 = 0,006)": [f for f in BASE if f != "log_competition"],
    "Sans coef_variation\n(corr. 0,82 avec uplift)": [f for f in BASE if f != "coef_variation"],
    "Sans uplift_promo": [f for f in BASE if f != "uplift_promo"],
    "Sans ratio_samedi_hp": [f for f in BASE if f != "ratio_samedi_hp"],
    "Sans les 3 variables\nde promo (CV, uplift, ratio)": [
        "log_ca_median", "log_competition", "ouvert_dimanche"
    ],
    "Avec Promo2\n(administratif)": BASE + ["Promo2"],
    "Avec StoreType one-hot\n(4 indicatrices)": BASE + [
        "type_a", "type_b", "type_c", "type_d"
    ],
}

resultats = []
for nom, feats in variantes.items():
    lab, sil = segmenter(feats)
    ari = adjusted_rand_score(labels_ref, lab)
    tailles = np.bincount(lab, minlength=K)
    resultats.append({
        "variante": nom,
        "n_variables": len(feats),
        "silhouette": sil,
        "ARI_vs_reference": ari,
        "tailles": " ".join(str(t) for t in tailles),
    })
    print(f"{nom.replace(chr(10), ' '):<45} sil={sil:.3f}  ARI={ari:.3f}  {resultats[-1]['tailles']}")

tableau = pd.DataFrame(resultats)

# ================================================================
# 2. Figure
# ================================================================
fig, ax = plt.subplots(figsize=(10, 0.62 * len(tableau) + 1.8))

noms_courts = [v.replace("\n", " ") for v in tableau["variante"]]
valeurs = tableau["ARI_vs_reference"].to_numpy()
couleurs = [
    "#2e7d32" if a >= 0.80 else ("#f9a825" if a >= 0.60 else "#c00000")
    for a in valeurs
]

barres = ax.barh(range(len(tableau)), valeurs, color=couleurs, alpha=0.9)
ax.set_yticks(range(len(tableau)))
ax.set_yticklabels(noms_courts, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("ARI avec la segmentation de référence")
ax.set_title("Analyse de sensibilité : le découpage survit-il au retrait d'une variable ?\n"
             "vert = conclusion inchangée (≥ 0,80) · jaune = proche · rouge = variable structurante")
ax.axvline(0.80, color="#2e7d32", linestyle="--", alpha=0.6)
ax.axvline(0.60, color="#f9a825", linestyle="--", alpha=0.6)

for barre, valeur in zip(barres, valeurs):
    ax.text(valeur + 0.015, barre.get_y() + barre.get_height() / 2,
            f"{valeur:.2f}", va="center", fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "06_sensibilite.png", bbox_inches="tight")
plt.close(fig)

# ================================================================
# 3. Lecture
# ================================================================
print("\n=== Lecture ===")
print("Référence : silence =", round(sil_ref, 3))
for ligne in resultats[1:]:
    ari = ligne["ARI_vs_reference"]
    if ari >= 0.80:
        verdict = "conclusion inchangée sans cette variable"
    elif ari >= 0.60:
        verdict = "structure proche, quelques magasins changent de segment"
    else:
        verdict = "cette variable porte réellement la séparation"
    print(f"  {ligne['variante'].replace(chr(10), ' '):<45} -> {verdict}")

print("\nFigure écrite : figures/06_sensibilite.png")
