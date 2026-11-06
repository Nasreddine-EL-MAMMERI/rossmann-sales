from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "train.csv", parse_dates=["Date"])
store = pd.read_csv(ROOT / "store.csv")
test = pd.read_csv(ROOT / "test.csv", parse_dates=["Date"])

df = train.merge(
    store[["Store", "StoreType"]],
    on="Store",
    how="left",
    validate="many_to_one"
)

print("Train :", train.shape, "| magasins :", train["Store"].nunique())
print("Après jointure avec store :", df.shape)
print("Dates train :", df["Date"].min().date(), "→", df["Date"].max().date())
print("Dates test :", test["Date"].min().date(), "→", test["Date"].max().date())

print("\nOpen dans train :")
print(df["Open"].value_counts(dropna=False))

print("\nOpen dans test :")
print(test["Open"].value_counts(dropna=False))

print("\nValeurs manquantes dans les colonnes utiles :")
print(df[["Sales", "Open", "Promo", "DayOfWeek", "StoreType"]].isna().sum())

#validation temporelle
date_split = pd.Timestamp("2015-06-14")

train_part = df[df["Date"] < date_split].copy()
valid_part = df[df["Date"] >= date_split].copy()

print("\n--- Split temporel ---")
print(
    "Train :",
    train_part["Date"].min().date(),
    "→",
    train_part["Date"].max().date(),
    "| lignes :",
    len(train_part)
)
print(
    "Validation :",
    valid_part["Date"].min().date(),
    "→",
    valid_part["Date"].max().date(),
    "| lignes :",
    len(valid_part)
)

print("\nOpen dans la validation :")
print(valid_part["Open"].value_counts(dropna=False))

print(
    "\nLignes de validation avec Sales > 0 :",
    int((valid_part["Sales"] > 0).sum())
)

#NAN dans open
open_manquant = test.loc[
    test["Open"].isna(),
    ["Id", "Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]
]

print("\nLignes du test où Open est manquant :")
print(open_manquant.to_string(index=False))

#magasin_622
print("\nTest — toutes les lignes du magasin 622 :")
print(
    test.loc[
        test["Store"] == 622,
        ["Id", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    ].sort_values("Date").to_string(index=False)
)

historique_622 = train.loc[train["Store"] == 622].copy()

print("\nHistorique du magasin 622 — ouvertures par jour de semaine :")
print(
    historique_622.groupby("DayOfWeek")["Open"]
    .agg(["count", "sum", "mean"])
)

print("\nDernières lignes du train pour le magasin 622 :")
print(
    historique_622.sort_values("Date").tail(20)[
        ["Date", "DayOfWeek", "Open", "Sales", "Promo"]
    ].to_string(index=False)
)
#__les jours fermés ne servent pas à apprendre les ventes des jours ouverts__
train_fit = train_part.loc[
    (train_part["Open"] == 1) & (train_part["Sales"] > 0)
].copy()

valid_eval = valid_part.loc[
    (valid_part["Open"] == 1) & (valid_part["Sales"] > 0)
].copy()

print("\n--- Lignes utilisées pour le modèle ---")
print("Entraînement :", len(train_fit))
print("Validation pour le RMSPE :", len(valid_eval))
print(
    "Validation : lignes ouvertes avec Sales = 0 :",
    int(((valid_part["Open"] == 1) & (valid_part["Sales"] == 0)).sum())
)

#moyenne par magasin pour la comparaison 
moyenne_par_magasin = train_fit.groupby("Store")["Sales"].mean()

y_true = valid_eval["Sales"].to_numpy()
y_pred = valid_eval["Store"].map(moyenne_par_magasin).to_numpy()

print("\n--- Baseline : moyenne historique par magasin ---")
print("Prédictions manquantes :", int(pd.isna(y_pred).sum()))

rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"RMSPE : {rmspe:.2%}")
print(f"RMSE  : {rmse:.2f} €")


#MODELE RIDGE 
colonnes = [
    "Store", "StoreType", "DayOfWeek", "Promo",
    "StateHoliday", "SchoolHoliday"
]

X_train = train_fit[colonnes].copy()
X_valid = valid_eval[colonnes].copy()

X_train["Year"] = train_fit["Date"].dt.year.astype(str)
X_valid["Year"] = valid_eval["Date"].dt.year.astype(str)

X_train["Month"] = train_fit["Date"].dt.month.astype(str)
X_valid["Month"] = valid_eval["Date"].dt.month.astype(str)

colonnes_categorielles = colonnes + ["Year", "Month"]

for col in colonnes_categorielles:
    X_train[col] = X_train[col].astype(str)
    X_valid[col] = X_valid[col].astype(str)

modele_ridge = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr")
)

modele_ridge.fit(X_train, np.log1p(train_fit["Sales"]))

pred_ridge = np.maximum(
    np.expm1(modele_ridge.predict(X_valid)),
    0
)

y_valid = valid_eval["Sales"].to_numpy()

rmspe_ridge = np.sqrt(np.mean(((y_valid - pred_ridge) / y_valid) ** 2))
rmse_ridge = np.sqrt(np.mean((y_valid - pred_ridge) ** 2))

print("\n--- Ridge ---")
print(f"RMSPE : {rmspe_ridge:.2%} (baseline : {rmspe:.2%})")
print(f"RMSE  : {rmse_ridge:.2f} € (baseline : {rmse:.2f} €)")

#le model ameliore rmspe mais pas trop on va diagnàstiquer ou il se trompe 
diagnostic = valid_eval[
    ["StoreType", "DayOfWeek", "Promo", "Sales"]
].copy()

diagnostic["Prediction"] = pred_ridge
diagnostic["erreur_pct_au_carre"] = (
    (diagnostic["Sales"] - diagnostic["Prediction"]) / diagnostic["Sales"]
) ** 2

for colonne in ["Promo", "DayOfWeek", "StoreType"]:
    resume = diagnostic.groupby(colonne).agg(
        nb_journees_magasin=("Sales", "size"),
        ventes_moyennes=("Sales", "mean"),
        erreur_pct_au_carre_moyenne=("erreur_pct_au_carre", "mean")
    )

    resume["RMSPE_%"] = np.sqrt(
        resume.pop("erreur_pct_au_carre_moyenne")
    ) * 100

    print(f"\n--- Erreurs par {colonne} ---")
    print(resume.round(2).to_string())

    diagnostic["biais_relatif"] = (
    diagnostic["Prediction"] - diagnostic["Sales"]
) / diagnostic["Sales"]

croisement = diagnostic.groupby(["DayOfWeek", "Promo"]).agg(
    nb_journees=("Sales", "size"),
    ventes_reelles_moyennes=("Sales", "mean"),
    predictions_moyennes=("Prediction", "mean"),
    biais_relatif_moyen=(
        "biais_relatif",
        lambda x: 100 * x.mean()
    ),
    erreur_pct_au_carre_moyenne=("erreur_pct_au_carre", "mean")
)

croisement["RMSPE_%"] = (
    np.sqrt(croisement.pop("erreur_pct_au_carre_moyenne")) * 100
)

print("\n--- Erreurs par jour et promo ---")
print(croisement.round(2).to_string())

#l'effet jour de semaineXpromo ensemble pas séparés 
X_train_jp = X_train.copy()
X_valid_jp = X_valid.copy()

X_train_jp["Jour_Promo"] = (
    train_fit["DayOfWeek"].astype(str)
    + "_"
    + train_fit["Promo"].astype(str)
)

X_valid_jp["Jour_Promo"] = (
    valid_eval["DayOfWeek"].astype(str)
    + "_"
    + valid_eval["Promo"].astype(str)
)

modele_jour_promo = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr")
)

modele_jour_promo.fit(
    X_train_jp,
    np.log1p(train_fit["Sales"])
)

pred_jour_promo = np.maximum(
    np.expm1(modele_jour_promo.predict(X_valid_jp)),
    0
)

y_valid = valid_eval["Sales"].to_numpy()

rmspe_jour_promo = np.sqrt(
    np.mean(((y_valid - pred_jour_promo) / y_valid) ** 2)
)
rmse_jour_promo = np.sqrt(
    np.mean((y_valid - pred_jour_promo) ** 2)
)

print("\n--- Comparaison ---")
print(f"Ridge actuel       : {rmspe_ridge:.2%} | {rmse_ridge:.2f} €")
print(f"Ridge + Jour_Promo : {rmspe_jour_promo:.2%} | {rmse_jour_promo:.2f} €")

comparaison_erreurs = valid_eval[["DayOfWeek", "Promo", "Sales"]].copy()

comparaison_erreurs["err2_ridge"] = (
    (comparaison_erreurs["Sales"].to_numpy() - pred_ridge)
    / comparaison_erreurs["Sales"].to_numpy()
) ** 2

comparaison_erreurs["err2_jour_promo"] = (
    (comparaison_erreurs["Sales"].to_numpy() - pred_jour_promo)
    / comparaison_erreurs["Sales"].to_numpy()
) ** 2

resume_comparaison = comparaison_erreurs.groupby(
    ["DayOfWeek", "Promo"]
).agg(
    nb_journees=("Sales", "size"),
    rmspe_ridge=(
        "err2_ridge",
        lambda x: 100 * np.sqrt(x.mean())
    ),
    rmspe_jour_promo=(
        "err2_jour_promo",
        lambda x: 100 * np.sqrt(x.mean())
    )
)

print("\n--- RMSPE par jour et promo : avant / après interaction ---")
print(resume_comparaison.round(2).to_string())

#erreurs par magasin

diag_type_jour = valid_eval[
    ["DayOfWeek", "StoreType", "Sales"]
].copy()

diag_type_jour["Prediction"] = pred_jour_promo

diag_type_jour["erreur_pct_au_carre"] = (
    (diag_type_jour["Sales"] - diag_type_jour["Prediction"])
    / diag_type_jour["Sales"]
) ** 2

diag_type_jour["biais_relatif"] = (
    diag_type_jour["Prediction"] - diag_type_jour["Sales"]
) / diag_type_jour["Sales"]

resume_type_jour = diag_type_jour.groupby(
    ["DayOfWeek", "StoreType"]
).agg(
    nb_journees=("Sales", "size"),
    ventes_moyennes=("Sales", "mean"),
    predictions_moyennes=("Prediction", "mean"),
    biais_pct=("biais_relatif", lambda x: 100 * x.mean()),
    rmspe_pct=(
        "erreur_pct_au_carre",
        lambda x: 100 * np.sqrt(x.mean())
    )
)

print("\n--- Erreurs par jour et type de magasin ---")
print(resume_type_jour.round(2).to_string())

#modele avec stortype ft dayofweek et promo
X_train_type_jour = X_train_jp.copy()
X_valid_type_jour = X_valid_jp.copy()

X_train_type_jour["Type_Jour"] = (
    train_fit["StoreType"].astype(str)
    + "_"
    + train_fit["DayOfWeek"].astype(str)
)

X_valid_type_jour["Type_Jour"] = (
    valid_eval["StoreType"].astype(str)
    + "_"
    + valid_eval["DayOfWeek"].astype(str)
)

modele_type_jour = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr")
)

modele_type_jour.fit(
    X_train_type_jour,
    np.log1p(train_fit["Sales"])
)

pred_type_jour = np.maximum(
    np.expm1(modele_type_jour.predict(X_valid_type_jour)),
    0
)

rmspe_type_jour = np.sqrt(
    np.mean(((y_valid - pred_type_jour) / y_valid) ** 2)
)
rmse_type_jour = np.sqrt(
    np.mean((y_valid - pred_type_jour) ** 2)
)

print("\n--- Comparaison ---")
print(
    f"Meilleur actuel (Jour_Promo) : "
    f"{rmspe_jour_promo:.2%} | {rmse_jour_promo:.2f} €"
)
print(
    f"+ Type_Jour                  : "
    f"{rmspe_type_jour:.2%} | {rmse_type_jour:.2f} €"
)

#le dernier modele ameliore clairement RMSPE et RMSE verifions si il n'a pas deterioré les autres variables 

comparaison_type_jour = valid_eval[
    ["DayOfWeek", "StoreType", "Sales"]
].copy()

y = comparaison_type_jour["Sales"].to_numpy()

comparaison_type_jour["err2_avant"] = (
    (y - pred_jour_promo) / y
) ** 2

comparaison_type_jour["err2_apres"] = (
    (y - pred_type_jour) / y
) ** 2

resume = comparaison_type_jour.groupby(
    ["DayOfWeek", "StoreType"]
).agg(
    nb_journees=("Sales", "size"),
    rmspe_avant=("err2_avant", lambda x: 100 * np.sqrt(x.mean())),
    rmspe_apres=("err2_apres", lambda x: 100 * np.sqrt(x.mean()))
)

print("\n--- Effet de Type_Jour par jour et type ---")
print(resume.round(2).to_string())

# le meme modele sur la periode du 27 avril au 13 juin 2015 pour voir si il est stable sur la periode de test

def construire_features(part):
    X = part[
        ["Store", "StoreType", "DayOfWeek", "Promo",
         "StateHoliday", "SchoolHoliday"]
    ].copy()

    X["Year"] = part["Date"].dt.year.astype(str)
    X["Month"] = part["Date"].dt.month.astype(str)
    X["Jour_Promo"] = (
        part["DayOfWeek"].astype(str) + "_" + part["Promo"].astype(str)
    )
    X["Type_Jour"] = (
        part["StoreType"].astype(str) + "_" + part["DayOfWeek"].astype(str)
    )

    return X.astype(str)


debut_validation_2 = pd.Timestamp("2015-04-27")
fin_validation_2 = pd.Timestamp("2015-06-14")  

train_fit_2 = df.loc[
    (df["Date"] < debut_validation_2)
    & (df["Open"] == 1)
    & (df["Sales"] > 0)
].copy()

valid_eval_2 = df.loc[
    (df["Date"] >= debut_validation_2)
    & (df["Date"] < fin_validation_2)
    & (df["Open"] == 1)
    & (df["Sales"] > 0)
].copy()

X_train_2 = construire_features(train_fit_2)
X_valid_2 = construire_features(valid_eval_2)
y_valid_2 = valid_eval_2["Sales"].to_numpy()

# Baseline : moyenne historique par magasin
moyenne_magasin_2 = train_fit_2.groupby("Store")["Sales"].mean()
pred_baseline_2 = valid_eval_2["Store"].map(moyenne_magasin_2).to_numpy()

# Modèle retenu : Ridge avec les deux interactions
modele_2 = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr")
)

modele_2.fit(X_train_2, np.log1p(train_fit_2["Sales"]))
pred_modele_2 = np.maximum(
    np.expm1(modele_2.predict(X_valid_2)),
    0
)

def calculer_scores(y, predictions):
    rmspe = np.sqrt(np.mean(((y - predictions) / y) ** 2))
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    return rmspe, rmse

rmspe_base_2, rmse_base_2 = calculer_scores(y_valid_2, pred_baseline_2)
rmspe_modele_2, rmse_modele_2 = calculer_scores(y_valid_2, pred_modele_2)

print("\n--- Backtest temporel 2 ---")
print(
    "Validation :",
    valid_eval_2["Date"].min().date(),
    "→",
    valid_eval_2["Date"].max().date(),
    "| lignes :",
    len(valid_eval_2)
)
print(f"Baseline magasin : {rmspe_base_2:.2%} | {rmse_base_2:.2f} €")
print(f"Ridge interactions : {rmspe_modele_2:.2%} | {rmse_modele_2:.2f} €")
# le modele est stable sur d'autres periode futures 

##réentraîner sur tout le train et prédire le test
#open manquantes sont imputées à 1 (ouvert)  
# Réentraînement du modèle retenu sur toutes les ventes ouvertes positives
train_final = df.loc[
    (df["Open"] == 1) & (df["Sales"] > 0)
].copy()

X_final = construire_features(train_final)
y_final = np.log1p(train_final["Sales"])

modele_final = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr")
)

modele_final.fit(X_final, y_final)

# Ajouter StoreType au test
test_final = test.merge(
    store[["Store", "StoreType"]],
    on="Store",
    how="left",
    validate="many_to_one"
)

assert len(test_final) == len(test)

# Hypothèse verifiée: les Open manquants concernent Store 622,
# dont l'historique indique une ouverture les jours concernés.
open_manquant = test_final["Open"].isna()

if open_manquant.any():
    if not test_final.loc[open_manquant, "Store"].eq(622).all():
        raise ValueError("Valeur Open manquante pour un autre magasin.")

    test_final.loc[open_manquant, "Open"] = 1

test_final["Open"] = test_final["Open"].astype("int8")

# Les jours fermés ont Sales=0 ; le modèle prédit uniquement les jours ouverts
test_final["Sales"] = 0.0
masque_ouvert = test_final["Open"] == 1

X_test_ouvert = construire_features(
    test_final.loc[masque_ouvert].copy()
)

test_final.loc[masque_ouvert, "Sales"] = np.maximum(
    np.expm1(modele_final.predict(X_test_ouvert)),
    0
)

# Fichier de prédictions (pour Kaggle)
submission = (
    test_final[["Id", "Sales"]]
    .sort_values("Id")
    .copy()
)

submission.to_csv(ROOT / "submission.csv", index=False)

print("\n--- Fichier de prédictions ---")
print("Lignes :", len(submission))
print("Open manquants après traitement :", int(test_final["Open"].isna().sum()))
print("Sales manquants :", int(submission["Sales"].isna().sum()))
print("Open=0 avec Sales non nulle :", int(
    ((test_final["Open"] == 0) & (test_final["Sales"] != 0)).sum()
))
print(submission.head().to_string(index=False))
