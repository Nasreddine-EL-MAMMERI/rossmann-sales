# Rossmann Store Sales — Résultats de la partie supervisée

## Objectif

Prédire `Sales` (chiffre d'affaires journalier d'un magasin) sur la période
du 1er août au 17 septembre 2015, à partir de l'historique
du 1er janvier 2013 au 31 juillet 2015.

## Protocole

- **Métrique principale** : RMSPE (Root Mean Squared Percentage Error),
  la métrique officielle du concours.
- **Métrique secondaire** : RMSE (en euros).
- **Cible transformée** : `log1p(Sales)`, pour stabiliser la variance
  et limiter le poids des très gros magasins.
- **Validation temporelle** (jamais aléatoire) : deux fenêtres futures de
  48 jours chacune, soit la durée exacte de la période de test.
  - Fenêtre 1 : entraînement jusqu'au 13/06/2015, validation 14/06 → 31/07/2015.
  - Fenêtre 2 : entraînement jusqu'au 26/04/2015, validation 27/04 → 13/06/2015.
- **Périmètre d'entraînement** : jours ouverts (`Open = 1`) avec ventes
  strictement positives. Les jours fermés sont prédits à 0 par convention.
- **Variable exclue volontairement** : `Customers`. Elle est indisponible
  dans `test.csv`, donc inutilisable en production.

## Résultats

### Fenêtre de validation 1 (14/06 → 31/07/2015)

| Modèle | RMSPE | RMSE |
|---|---:|---:|
| Baseline : moyenne historique par magasin | 31,82 % | 1 960,23 € |
| Ridge one-hot (effets principaux) | 21,11 % | 1 345,87 € |
| + interaction `DayOfWeek × Promo` | 20,67 % | 1 303,81 € |
| **+ interaction `StoreType × DayOfWeek`** | **19,14 %** | **1 260,90 €** |

### Fenêtre de validation 2 (27/04 → 13/06/2015)

| Modèle | RMSPE | RMSE |
|---|---:|---:|
| Baseline : moyenne historique par magasin | 29,43 % | 1 903,02 € |
| **Ridge retenu (2 interactions)** | **19,67 %** | **1 423,03 €** |

Le modèle retenu est stable sur les deux fenêtres (19,14 % et 19,67 %),
alors que les baselines varient peu. L'écart au baseline est donc
structurel et non un artefact de découpage.

### Score Kaggle (données de test réelles)

| Classement | RMSPE |
|---|---:|
| Public | 0,17005 |
| **Privé** | **0,17384** |

Écart public/privé : 0,0038 — pas de sur-ajustement à une sous-période.

### Modèle retenu

```python
make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(alpha=10.0, solver="lsqr"),
)
```

Variables : `Store`, `StoreType`, `Assortment`, `DayOfWeek`, `Promo`,
`StateHoliday`, `SchoolHoliday`, `Year`, `Month`, plus les interactions
`Jour_Promo` (`DayOfWeek × Promo`) et `Type_Jour` (`StoreType × DayOfWeek`).

Entraîné sur `log1p(Sales)`, prédictions reconverties par `expm1` et bornées à 0.

## Hypothèse documentée : `Open` manquant dans le test

11 lignes de `test.csv` ont une valeur `Open` manquante, toutes pour le
magasin 622, du 5 au 17 septembre 2015 (hors dimanches, où `Open = 0` est connu).

Décision : **`Open = 1`** à la prédiction, sur la base de trois éléments :
- le taux d'ouverture historique du magasin 622 est de 94 à 100 % du lundi au samedi ;
- toutes ses dates de test connues avant le 5 septembre sont ouvertes ;
- les 11 dates concernées ne sont ni jours fériés ni vacances scolaires.

`test.csv` n'a pas été modifié : l'hypothèse est appliquée en mémoire au moment
de la prédiction.

## Limites à assumer dans le README

- Un **seul modèle global** : pas de modèle par magasin, alors que les
  solutions du haut de classement en utilisent.
- **Modèle linéaire** : les non-linéarités (effets de seuil, saisonnalité fine,
  jours fériés locaux) ne sont captées que partiellement par les interactions.
- **Pas de variables externes** : météo, jours fériés régionaux, état des
  concurrents ne sont pas exploités.
- **Erreurs concentrées sur les cas rares** : le samedi et le dimanche restent
  les journées les plus difficiles (RMSPE par groupe supérieur à 30 %),
  ce qui est cohérent avec leur faible volume d'entraînement.
- Le score de validation ne prédit pas exactement le score test : ici il était
  pessimiste de 2 points, ce qui illustre l'incertitude du protocole.

## Pistes d'amélioration (non poursuivies)

1. Un modèle par type de magasin ou par magasin, avec régularisation.
2. Gradient boosting (LightGBM, XGBoost) sur les mêmes variables.
3. Pondération des erreurs par le chiffre d'affaires réel, pour coller
   exactement à la métrique RMSPE.
4. Traitement spécifique des fermetures exceptionnelles et des promotions
   prolongées (`Promo2`).
