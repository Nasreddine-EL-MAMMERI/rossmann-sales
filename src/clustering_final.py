"""Segmentation finale : correction de la redondance + test de stabilité.

Corrections par rapport à la v4 :
  1. ratio_samedi est recalculé SANS les jours de promotion, des deux côtés
     (samedi et semaine). En v4, il mesurait l'élasticité promotionnelle
     deux fois, d'où deux eta2 identiques à 0.542.
  2. Test de stabilité : rééchantillonnage à 80 % des magasins, 30 tirages,
     indice ARI sur les magasins laissés de côté.
  3. Variables finales (5) après analyse de sensibilité :
     log_ca_median, coef_variation, ouvert_dimanche, uplift_promo,
     ratio_samedi_hp. log_competition retirée (eta2 = 0.006, ARI = 0.969,
     silhouette 0.296 -> 0.360) mais conservée comme variable descriptive.
  4. Figures régénérées avec les noms des segments.

"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

COULEURS = plt.get_cmap("tab10").colors
RNG = np.random.default_rng(42)


def sauvegarder(fig, nom):
    chemin = FIG / nom
    try:
        fig.savefig(chemin, bbox_inches="tight")
        print(f"  figure écrite : {nom}")
    except OSError as erreur:
        secours = FIG / f"{chemin.stem}_nouveau.png"
        try:
            fig.savefig(secours, bbox_inches="tight")
            print(f"  /!\\ {nom} verrouillé ({erreur}).")
            print(f"      Écrit sous : {secours.name} — ferme la visionneuse, "
                  "puis renomme ou relance.")
        except OSError:
            print(f"  /!\\ Écriture impossible pour {nom} ({erreur}).")
    finally:
        plt.close(fig)


# ================================================================
# 1. Variables
# ================================================================
profil = pd.read_csv(ROOT / "profil_magasin_prepare.csv")

train = pd.read_csv(
    ROOT / "train.csv",
    usecols=["Store", "DayOfWeek", "Sales", "Promo", "Open", "Customers"],
)
ouvert = train.loc[(train["Open"] == 1) & (train["Sales"] > 0)]
hors_promo = ouvert.loc[ouvert["Promo"] == 0]

# Élasticité promotionnelle
uplift = (
    ouvert.loc[ouvert["Promo"] == 1].groupby("Store")["Sales"].mean()
    / hors_promo.groupby("Store")["Sales"].mean()
).rename("uplift_promo")

# Effet week-end, promotions neutralisées des deux côtés
ratio_samedi = (
    hors_promo.loc[hors_promo["DayOfWeek"] == 6].groupby("Store")["Sales"].mean()
    / hors_promo.loc[hors_promo["DayOfWeek"].isin([1, 2, 3, 4, 5])]
      .groupby("Store")["Sales"].mean()
).rename("ratio_samedi_hp")

nouvelles = pd.concat([uplift, ratio_samedi], axis=1).reset_index()
profil = profil.merge(nouvelles, on="Store", how="left", validate="one_to_one")

print("=== Valeurs manquantes sur les nouvelles variables ===")
print(profil[["uplift_promo", "ratio_samedi_hp"]].isna().sum().to_string())
for col in ["uplift_promo", "ratio_samedi_hp"]:
    profil[col] = profil[col].fillna(profil[col].median())

# log_competition retirée après l'analyse de sensibilité (clustering_sensibilite.py) :
#   eta2 = 0.006, ARI sans elle = 0.969, silhouette 0.296 -> 0.360.
# Elle reste utilisée comme variable DESCRIPTIVE des segments....................

features = [
    "log_ca_median", "coef_variation",
    "ouvert_dimanche", "uplift_promo", "ratio_samedi_hp",
]
LIBELLES = {
    "log_ca_median": "CA médian (log)",
    "coef_variation": "Variabilité (CV)",
    "log_competition": "Distance concurrent (log)",
    "ouvert_dimanche": "Ouvert dimanche",
    "uplift_promo": "Sensibilité promo",
    "ratio_samedi_hp": "Ratio samedi/semaine (hors promo)",
}

X = profil[features]
print("\n=== Corrélations (contrôle de redondance) ===")
print(X.corr().round(2).to_string())

X_scaled = StandardScaler().fit_transform(X)

# ================================================================
# 2. Choix de k
# ================================================================
ks = list(range(2, 11))
inerties, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X_scaled)
    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lab))

K = 3  # choix argumenté : vous pouvez lire les arguments dans le README 
print(f"\n=== Choix de k ===")
for k, i, s in zip(ks, inerties, silhouettes):
    marque = "  <-- retenu" if k == K else ""
    print(f"  k={k:2d} : inertie={i:8.1f}  silhouette={s:.3f}{marque}")

# ================================================================
# 3. Ajustement final
# ================================================================
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
profil["cluster"] = labels

resume = profil.groupby("cluster").agg(
    effectif=("Store", "size"),
    ca_median_eur=("log_ca_median", lambda s: int(np.expm1(s).median())),
    coef_variation=("coef_variation", "mean"),
    part_dimanche=("ouvert_dimanche", "mean"),
    competition_m=("log_competition", lambda s: int(np.expm1(s).median())),
    uplift_promo=("uplift_promo", "mean"),
    ratio_samedi_hp=("ratio_samedi_hp", "mean"),
).round(3)

print("\n=== Profils en unités réelles ===")
print(resume.to_string())

# --- étiquettes métier automatiques
noms = {}
for c in resume.index:
    if resume.loc[c, "part_dimanche"] > 0.5:
        noms[c] = "Proximité (dimanche)"
    elif resume.loc[c, "uplift_promo"] > resume["uplift_promo"].median():
        noms[c] = "Sensible aux promotions"
    else:
        noms[c] = "Standard / régulier"
print("\n=== Noms proposés ===")
for c, n in noms.items():
    print(f"  Segment {c} -> {n} (n={resume.loc[c, 'effectif']})")

# ================================================================
# 4. eta2
# ================================================================
print("\n=== eta2 : quelle variable porte la séparation ? ===")
df_scaled = pd.DataFrame(X_scaled, columns=features)
for col in features:
    moy = df_scaled[col].mean()
    ss_tot = ((df_scaled[col] - moy) ** 2).sum()
    ss_int = sum(len(g) * (g[col].mean() - moy) ** 2
                 for _, g in df_scaled.groupby(labels))
    print(f"  {col:<20} eta2 = {ss_int / ss_tot:.3f}")

# ================================================================
# 5. Stabilite (ARI sur echantillons de 80 %)
# ================================================================
print("\n=== Test de stabilite (30 rééchantillonnages à 80 %) ===")
n = len(X_scaled)
aris = []
for b in range(30):
    idx = RNG.choice(n, size=int(0.8 * n), replace=False)
    km_b = KMeans(n_clusters=K, random_state=b, n_init=10).fit(X_scaled[idx])
    lab_b = km_b.predict(X_scaled)
    reste = np.setdiff1d(np.arange(n), idx)
    aris.append(adjusted_rand_score(labels[reste], lab_b[reste]))

aris = np.array(aris)
print(f"  ARI moyen : {aris.mean():.3f} | médian : {np.median(aris):.3f} "
      f"| min : {aris.min():.3f} | max : {aris.max():.3f}")
print("  Lecture : > 0.75 = stable ; 0.60-0.75 = acceptable ; < 0.60 = instable")

# ================================================================
# 6. Validations externes
# ================================================================
modalite_type = np.select(
    [profil["type_a"] == 1, profil["type_b"] == 1,
     profil["type_c"] == 1, profil["type_d"] == 1],
    ["a", "b", "c", "d"], default="?",
)

print("\n=== Validation externe : StoreType (% par ligne) ===")
print((pd.crosstab(profil["cluster"], modalite_type, normalize="index") * 100)
      .round(1).to_string())

print("\n=== Validation externe : Promo2 (moyenne) ===")
print(profil.groupby("cluster")["Promo2"].mean().round(3).to_string())

ouverts_c = train.loc[(train["Open"] == 1) & (train["Customers"] > 0)].copy()
ouverts_c["panier"] = ouverts_c["Sales"] / ouverts_c["Customers"]
trafic = ouverts_c.groupby("Store").agg(
    clients_jour=("Customers", "mean"), panier_moyen=("panier", "mean"))
profil = profil.merge(trafic, on="Store", how="left", validate="one_to_one")

print("\n=== Validation externe : trafic et panier ===")
print(profil.groupby("cluster")[["clients_jour", "panier_moyen"]]
      .mean().round(2).to_string())

# ================================================================
# 7. FIGURES
# ================================================================
# --- 1. choix de k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(ks, inerties, "o-", color="#1f4e79", linewidth=2, markersize=7)
ax1.axvline(K, linestyle="--", color="#c00000", alpha=0.7, label=f"k retenu = {K}")
ax1.set_xlabel("Nombre de clusters (k)")
ax1.set_ylabel("Inertie intra-classe")
ax1.set_title("Méthode du coude")
ax1.legend(frameon=False)

ax2.plot(ks, silhouettes, "s-", color="#2e7d32", linewidth=2, markersize=7)
ax2.axvline(K, linestyle="--", color="#c00000", alpha=0.7, label=f"k retenu = {K}")
ax2.set_xlabel("Nombre de clusters (k)")
ax2.set_ylabel("Silhouette moyenne")
ax2.set_title("Score de silhouette")
ax2.legend(frameon=False)
for ax, vals in ((ax1, inerties), (ax2, silhouettes)):
    for x, y in zip(ks, vals):
        ax.annotate(f"{y:.2f}" if ax is ax2 else f"{y:.0f}", (x, y),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#444")
fig.tight_layout()
sauvegarder(fig, "01_choix_k.png")

# --- 2. ACP
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8.5, 6))
for c in range(K):
    m = labels == c
    ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.6,
               color=COULEURS[c], label=f"{noms[c]} (n={m.sum()})")
    cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
    ax.scatter(cx, cy, marker="X", s=240, color=COULEURS[c],
               edgecolor="black", linewidth=1.2, zorder=5)
    ax.annotate(str(c), (cx, cy), fontsize=11, fontweight="bold",
                ha="center", va="center", color="white", zorder=6)

ax.set_xlabel(f"Composante 1 ({var[0]:.0f} % de la variance)")
ax.set_ylabel(f"Composante 2 ({var[1]:.0f} % de la variance)")
ax.set_title("Segmentation des 1 115 magasins (ACP)\n"
             f"Les 2 axes résument {var.sum():.0f} % de l'information")
ax.legend(frameon=True, fontsize=8)
fig.tight_layout()
sauvegarder(fig, "02_acp_clusters.png")

# --- 3. heatmap
centres_std = pd.DataFrame(X_scaled, columns=features).groupby(labels).mean()
fig, ax = plt.subplots(figsize=(10, 1.2 * K + 2))
im = ax.imshow(centres_std.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(features)))
ax.set_xticklabels([LIBELLES[f] for f in features], rotation=25, ha="right")
ax.set_yticks(range(K))
ax.set_yticklabels([f"{c} — {noms[c]}" for c in range(K)])
ax.set_title("Profil de chaque segment (écarts à la moyenne, en σ)")
ax.grid(False)
for i in range(K):
    for j in range(len(features)):
        v = centres_std.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v) > 1.4 else "black")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("écart à la moyenne (σ)")
fig.tight_layout()
sauvegarder(fig, "03_heatmap_profils.png")

# --- 4. distributions
n_cols = 3
n_lignes = int(np.ceil(len(features) / n_cols))
fig, axes = plt.subplots(n_lignes, n_cols,
                         figsize=(4.3 * n_cols, 3.5 * n_lignes))
axes = np.atleast_1d(axes).ravel()

# On masque les panneaux inutilisés si le nombre de variables n'est pas un multiple du nombre de colonnes.
for ax_vide in axes[len(features):]:
    ax_vide.set_visible(False)

for j, col in enumerate(features):
    ax = axes[j]
    bp = ax.boxplot([X_scaled[labels == c, j] for c in range(K)],
                    patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], range(K)):
        patch.set_facecolor(COULEURS[c])
        patch.set_alpha(0.75)
    ax.set_title(LIBELLES[col], fontsize=10)
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([str(c) for c in range(K)])
    ax.set_xlabel("segment")
    ax.set_ylabel("valeur standardisée")
fig.suptitle("Distribution des variables par segment", fontsize=12)
fig.tight_layout()
sauvegarder(fig, "04_distributions.png")

# --- 5. stabilité
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(aris, bins=12, color="#1f4e79", alpha=0.85, edgecolor="white")
ax.axvline(aris.mean(), color="#c00000", linestyle="--",
           label=f"ARI moyen = {aris.mean():.2f}")
ax.set_xlabel("ARI entre le découpage de référence et un rééchantillonnage")
ax.set_ylabel("Nombre de tirages")
ax.set_title("Stabilité de la segmentation (30 rééchantillonnages à 80 %)")
ax.legend(frameon=False)
fig.tight_layout()
sauvegarder(fig, "05_stabilite.png")

# ================================================================
# 8. Sortie finale
# ================================================================
profil["nom_segment"] = profil["cluster"].map(noms)
profil.to_csv(ROOT / "profil_magasin_clusters_final.csv", index=False)

print("\n=== Tableau de synthèse (pour le README) ===")
synthese = resume.copy()
synthese.insert(1, "nom", [noms[c] for c in synthese.index])
print(synthese.to_string())

print("\n=== Figures écrites ===")
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)

# ===========================
# 9. Fichier de contrôle 
# ===========================
sil_k = silhouettes[ks.index(K)]

controle = {
    "date_execution": datetime.now().isoformat(timespec="seconds"),
    "n_magasins": int(len(X)),
    "n_variables": len(features),
    "variables": features,
    "k": int(K),
    "silhouette_k": round(float(sil_k), 4),
    "inertie_k": round(float(inerties[ks.index(K)]), 2),
    "tailles_segments": {
        f"segment_{c}": int((labels == c).sum()) for c in range(K)
    },
    "noms_segments": {f"segment_{c}": noms[c] for c in range(K)},
    "ari_moyen": round(float(aris.mean()), 4),
    "ari_median": round(float(np.median(aris)), 4),
    "ari_min": round(float(aris.min()), 4),
    "clients_par_jour": {
        f"segment_{c}": round(float(v), 2)
        for c, v in profil.groupby("cluster")["clients_jour"].mean().items()
    },
    "panier_moyen": {
        f"segment_{c}": round(float(v), 2)
        for c, v in profil.groupby("cluster")["panier_moyen"].mean().items()
    },
    "figures": [
        {
            "fichier": f.name,
            "octets": f.stat().st_size,
            "modifie": datetime.fromtimestamp(f.stat().st_mtime).isoformat(
                timespec="seconds"),
        }
        for f in sorted(FIG.glob("*.png"))
    ],
}

chemin_controle = ROOT / "resultats_clustering.json"
with open(chemin_controle, "w", encoding="utf-8") as fh:
    json.dump(controle, fh, ensure_ascii=False, indent=2)

print("\n=== CONTRÔLE ===")
print(f"  {len(features)} variables, k = {K}, "
      f"silhouette = {sil_k:.4f}, inertie = {inerties[ks.index(K)]:.1f}")
print(f"  Tailles : {controle['tailles_segments']}")
print(f"  ARI moyen = {aris.mean():.3f} | médian = {np.median(aris):.3f}")
print(f"  Fichier de contrôle écrit : {chemin_controle.name}")
print("  Envoie ce fichier JSON : il contient tous les chiffres à vérifier.")

