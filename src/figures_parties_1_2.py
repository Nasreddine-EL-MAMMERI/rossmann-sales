"""
Régénère les figures de publication des parties 1 et 2.

Les analyses exploratoires (SQL, EDA) sont menées en interactif.
Ce script reconstruit les figures affichées dans le README à partir
des données brutes, afin qu'elles restent reproductibles.

Produit quatre figures, numérotées à la suite de celles du clustering :

  07_structure_reseau.png    CA total et CA moyen par magasin, par type
  08_promo_jour_semaine.png  ventes moyennes par jour de semaine, promo vs hors promo
  09_predit_vs_reel.png      nuage hexbin des prédictions contre les ventes réelles
  10_erreur_jour_semaine.png RMSPE par jour de semaine sur la fenêtre de validation

).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLEU = "#1f4e79"
VERT = "#2e7d32"
ORANGE = "#e07b00"
ROUGE = "#c00000"


def sauvegarder(fig, nom):
    """Écriture robuste : un fichier verrouillé ne doit pas tout interrompre."""
    chemin = FIG / nom
    try:
        fig.savefig(chemin, bbox_inches="tight")
        print(f"  figure écrite : {nom}")
    except OSError as erreur:
        secours = FIG / f"{chemin.stem}_nouveau.png"
        try:
            fig.savefig(secours, bbox_inches="tight")
            print(f"  /!\\ {nom} verrouillé ({erreur}). Écrit sous {secours.name}")
        except OSError:
            print(f"  /!\\ Écriture impossible pour {nom} ({erreur}).")
    finally:
        plt.close(fig)


# ================================================================
# LECTURE
# ================================================================
store = pd.read_csv(ROOT / "store.csv")

# Partie 1 : colonnes descriptives seulement
train_light = pd.read_csv(
    ROOT / "train.csv",
    usecols=["Store", "DayOfWeek", "Sales", "Promo", "Open"],
)

# Partie 2 : tout ce qui sert au modèle
train = pd.read_csv(
    ROOT / "train.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": str},
)

train = train.merge(
    store[["Store", "StoreType"]], on="Store", how="left", validate="many_to_one"
)

print("train :", train.shape)

# ================================================================
# FIGURE 07 — structure du réseau par type de magasin
# ================================================================
par_type = train_light.merge(
    store[["Store", "StoreType"]], on="Store", how="left", validate="many_to_one"
)

ca_total = par_type.groupby("StoreType")["Sales"].sum() / 1e6
nb_magasins = store["StoreType"].value_counts()
ca_moyen = (ca_total * 1e6 / nb_magasins.reindex(ca_total.index)) / 1e3

ordre = ["a", "d", "c", "b"]
ca_total = ca_total.reindex(ordre)
ca_moyen = ca_moyen.reindex(ordre)
effectifs = nb_magasins.reindex(ordre)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

barres = ax1.bar(ordre, ca_total.values, color=BLEU, alpha=0.85)
ax1.set_title("Chiffre d'affaires cumulé 2013–2015\npar type de magasin")
ax1.set_ylabel("Millions d'euros")
ax1.set_xlabel("Type de magasin")
for barre, valeur, n in zip(barres, ca_total.values, effectifs.values):
    ax1.text(barre.get_x() + barre.get_width() / 2, valeur + 30,
             f"{valeur:,.0f} M€\n({n} magasins)".replace(",", " "),
             ha="center", fontsize=8.5)

barres = ax2.bar(ordre, ca_moyen.values, color=ORANGE, alpha=0.85)
ax2.set_title("Chiffre d'affaires moyen par magasin\n(même période)")
ax2.set_ylabel("Milliers d'euros par magasin")
ax2.set_xlabel("Type de magasin")
for barre, valeur in zip(barres, ca_moyen.values):
    ax2.text(barre.get_x() + barre.get_width() / 2, valeur + 100,
             f"{valeur:,.0f} k€".replace(",", " "), ha="center", fontsize=8.5)

fig.suptitle("Le type b est le plus petit effectif et le plus fort CA moyen — "
             "un effet de composition, pas une preuve de rentabilité",
             fontsize=9.5, color="#444")
fig.tight_layout()
sauvegarder(fig, "07_structure_reseau.png")

print("\nCA total (M€) :", ca_total.round(1).to_dict())
print("CA moyen/magasin (k€) :", ca_moyen.round(0).to_dict())

# ================================================================
# FIGURE 08 — effet promotion selon le jour de la semaine
# ================================================================
ouverts = train_light.loc[train_light["Open"] == 1]
moyennes = (ouverts.groupby(["DayOfWeek", "Promo"])["Sales"]
            .mean().unstack("Promo"))

jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
positions = np.arange(len(moyennes))
largeur = 0.38

fig, ax = plt.subplots(figsize=(9.5, 4.6))

if 0 in moyennes.columns:
    ax.bar(positions - largeur / 2, moyennes[0].values, largeur,
           label="sans promotion", color="#8fa8bf")
if 1 in moyennes.columns:
    valeurs_promo = moyennes[1].reindex(moyennes.index)
    ax.bar(positions + largeur / 2, valeurs_promo.fillna(0).values, largeur,
           label="avec promotion", color=BLEU)

ax.set_xticks(positions)
# Les étiquettes suivent les jours réellement présents (le dimanche peut
# être absent si aucun magasin n'ouvre ce jour-là dans l'historique)
ax.set_xticklabels([jours[j - 1] for j in moyennes.index])
ax.set_ylabel("Ventes moyennes (€)")
ax.set_xlabel("Jour de la semaine")
ax.set_title("Ventes moyennes par jour, avec et sans promotion\n"
             "Les promotions n'ont lieu qu'en semaine : le week-end n'a pas "
             "de barre « avec promotion »")
ax.legend(frameon=False)

# Écart relatif affiché au-dessus de chaque paire complète
for i, jour in enumerate(moyennes.index):
    if 0 in moyennes.columns and 1 in moyennes.columns:
        sans = moyennes.loc[jour, 0]
        avec = moyennes.loc[jour, 1]
        if pd.notna(avec):
            ecart = (avec / sans - 1) * 100
            ax.annotate(f"+{ecart:.0f} %", (positions[i], max(sans, avec) + 400),
                        ha="center", fontsize=8.5, color=VERT, fontweight="bold")

fig.tight_layout()
sauvegarder(fig, "08_promo_jour_semaine.png")

print("\nVentes moyennes par jour et par promo :")
print(moyennes.round(0).to_string())

# ================================================================
# FIGURE 09 et 10 — modèle supervisé
# ================================================================
DEBUT_VALID = pd.Timestamp("2015-06-14")
FIN_VALID = pd.Timestamp("2015-07-31")

train_fit = train.loc[
    (train["Date"] < DEBUT_VALID) & (train["Open"] == 1) & (train["Sales"] > 0)
].copy()
valid = train.loc[
    (train["Date"] >= DEBUT_VALID) & (train["Date"] <= FIN_VALID)
    & (train["Open"] == 1)
].copy()

print(f"\nEntraînement : {len(train_fit):,} lignes | "
      f"validation : {len(valid):,} lignes".replace(",", " "))


def ajouter_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Jour_Promo"] = df["DayOfWeek"].astype(str) + "_" + df["Promo"].astype(str)
    df["Type_Jour"] = df["StoreType"].astype(str) + "_" + df["DayOfWeek"].astype(str)
    return df


train_fit = ajouter_features(train_fit)
valid = ajouter_features(valid)

CATEGORIES = [
    "Store", "StoreType", "DayOfWeek", "Promo", "StateHoliday",
    "SchoolHoliday", "Year", "Month", "Jour_Promo", "Type_Jour",
]

modele = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
    Ridge(alpha=10.0, solver="lsqr"),
)

modele.fit(train_fit[CATEGORIES], np.log1p(train_fit["Sales"]))

pred = np.maximum(np.expm1(modele.predict(valid[CATEGORIES])), 0)
reel = valid["Sales"].to_numpy()

erreur = (reel - pred) / reel
rmspe = np.sqrt(np.mean(erreur ** 2)) * 100
rmse = np.sqrt(np.mean((reel - pred) ** 2))

print(f"RMSPE validation : {rmspe:.2f} % | RMSE : {rmse:.2f} €")

# --- FIGURE 09 : prédit vs réel
fig, ax = plt.subplots(figsize=(7.6, 6.4))

hb = ax.hexbin(reel, pred, gridsize=60, bins="log", cmap="viridis", mincnt=1)
limite = max(reel.max(), pred.max()) * 1.02
ax.plot([0, limite], [0, limite], color=ROUGE, linestyle="--",
        linewidth=1.8, label="prédiction parfaite (y = x)")

ax.set_xlim(0, limite)
ax.set_ylim(0, limite)
ax.set_xlabel("Ventes réelles (€)")
ax.set_ylabel("Ventes prédites (€)")
ax.set_title("Prédictions contre ventes réelles — fenêtre de validation\n"
             f"14 juin → 31 juillet 2015 · {len(valid):,} jours-magasins".replace(",", " ")
             + f" · RMSPE {rmspe:.1f} %")
ax.legend(frameon=True, loc="upper left")
cbar = fig.colorbar(hb, ax=ax, label="nombre de jours-magasins (échelle log)")

fig.tight_layout()
sauvegarder(fig, "09_predit_vs_reel.png")

# --- FIGURE 10 : RMSPE par jour de semaine
valid = valid.assign(pred=pred, erreur_abs=erreur ** 2)
par_jour = valid.groupby("DayOfWeek").agg(
    rmspe=("erreur_abs", lambda s: np.sqrt(s.mean()) * 100),
    effectif=("Sales", "size"),
    ventes_moyennes=("Sales", "mean"),
)

fig, ax = plt.subplots(figsize=(9, 4.4))
couleurs = [ROUGE if valeur > 30 else (ORANGE if valeur > 20 else BLEU)
            for valeur in par_jour["rmspe"]]
barres = ax.bar([jours[j - 1] for j in par_jour.index], par_jour["rmspe"].values,
                color=couleurs, alpha=0.88)

for barre, (_, ligne) in zip(barres, par_jour.iterrows()):
    ax.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.6,
            f"{ligne['rmspe']:.1f} %\n(n={ligne['effectif']:.0f})",
            ha="center", fontsize=8.5)

ax.axhline(rmspe, color=VERT, linestyle="--", linewidth=1.6,
           label=f"RMSPE global = {rmspe:.1f} %")
ax.set_ylabel("RMSPE (%)")
ax.set_xlabel("Jour de la semaine")
ax.set_title("Erreur de prédiction par jour de semaine\n"
             "Le week-end concentre l'erreur : peu de jours dans l'historique, "
             "donc peu d'exemples pour le modèle")
ax.legend(frameon=False)
fig.tight_layout()
sauvegarder(fig, "10_erreur_jour_semaine.png")

print("\nRMSPE par jour de semaine :")
print(par_jour.round(2).to_string())

print("\nFigures écrites :")
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)
