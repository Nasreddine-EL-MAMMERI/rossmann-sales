# Journal des décisions — segmentation des magasins Rossmann

Chaque décision est justifiée par une mesure. Les pistes abandonnées sont
conservées : elles font partie du raisonnement et prouvent la démarche.

---

## D1 — Agréger sur les jours d'ouverture uniquement

**Décision** : les moyennes de ventes par magasin sont calculées sur
`Open == 1` et `Sales > 0`.

**Raison** : sur les 1 017 209 lignes d'entraînement, 172 817 sont des jours
fermés, soit 17 %. Les inclure aurait mesuré un taux de fermeture déguisé en
performance commerciale.

---

## D2 — Écarter `part_jours_promo` (quasi constante + artefact)

**Mesures** : moyenne 0,446, écart-type 0,010 → **2,3 % de variation relative**.
Corrélation avec `taux_ouverture_dimanche` : **−0,942**.

**Interprétation** : les promotions sont planifiées au niveau de l'enseigne, pas
par magasin. La corrélation extrême s'explique arithmétiquement : le taux est
calculé sur les jours d'ouverts, et le dimanche n'est jamais un jour de promo.
Un magasin fermé le dimanche affiche ≈ 0,45 ; s'il ouvre aussi le dimanche, sa
part tombe à 0,45 × 6/7 ≈ 0,386 — le minimum observé est 0,377.

**Décision** : variable supprimée. Après standardisation, ses micro-écarts
auraient été amplifiés à l'échelle du chiffre d'affaires.

---

## D3 — Remplacer `variabilite_ca` par le coefficient de variation

**Mesures** : corrélation variabilité / CA moyen = **0,774**. Corrélation
CV / CA médian = **−0,295**.

**Interprétation** : l'écart-type en euros croît mécaniquement avec la taille.
Le coefficient de variation (écart-type / moyenne) mesure l'irrégularité
**indépendamment du volume**.

**Décision** : `variabilite_ca` écartée, `coef_variation` retenue.

---

## D4 — Garder la médiane, écarter la moyenne

**Mesure** : corrélation `ca_moyen_jour` / `ca_median_jour` = **0,9974**.

Les deux variables sont interchangeables ; la médiane est insensible aux
journées exceptionnelles. Garder les deux aurait doublé le poids de la taille.

---

## D5 — Première tentative : échec par les indicatrices (v1)

**Configuration** : 12 dimensions dont **7 indicatrices** (`StoreType` ×4,
`Assortment` ×3), soit 58 % du poids géométrique.

**Symptômes observés** :
- tailles de clusters **17**, **148**, **348**, **9** — identiques aux effectifs
  des modalités de `StoreType` et `Assortment` ;
- silhouette croissante avec k (0,230 → 0,339), signature de l'isolement
  successif de petits groupes ;
- coude non monotone (−21 %, −13 %, −16 %, −10 %).

**Diagnostic** : KMeans redécouvrait une variable déjà présente dans le fichier
`store.csv`. Aucune information nouvelle.

**Décision** : les catégorielles sortent du clustering et deviennent des
variables de **validation externe**.

---

## D6 — Deuxième tentative : échec par une variable binaire (v2)

**Configuration** : 5 variables dont deux binaires (`ouvert_dimanche`, `Promo2`).

**Symptômes** : les clusters 0 et 1 étaient identiques sur tout sauf `Promo2`
(1,00 contre 0,00), avec 10 % d'écart de CA seulement. η² de `Promo2` ≈ 0,7.

**Interprétation** : un binaire équilibré 51/49 offre deux groupes à distance
maximale pour un coût géométrique nul. `Promo2` n'est pas un comportement mais
une adhésion administrative.

**Décision** : `Promo2` retirée du clustering, conservée en validation externe.

---

## D7 — Troisième tentative : variables de comportement (v3 / v4)

**Ajouts** :
- `uplift_promo` = ventes moyennes en promo / hors promo → élasticité ;
- `ratio_samedi` = ventes du samedi / ventes en semaine → profil hebdomadaire.

**Problème détecté en v4** : `uplift_promo` et `ratio_samedi` avaient un η²
**rigoureusement identique (0,542)**, signe de redondance. Mécanisme : les
promotions ont lieu en semaine, donc un magasin sensible aux promos voit son
ratio samedi/semaine baisser artificiellement.

**Correction** : `ratio_samedi_hp` recalculé **hors jours de promotion des deux
côtés**. Corrélation avec l'élasticité ramenée de −0,94 à −0,68.

---

## D8 — Analyse de sensibilité (décision finale sur les variables)

| Variante | Silhouette | ARI vs référence | Effet |
|---|---:|---:|---|
| Référence (6 variables) | 0,296 | 1,000 | — |
| **Sans `log_competition`** | **0,360** | 0,969 | redondante → **retirée** |
| Sans `coef_variation` | 0,279 | 0,757 | structurante → gardée |
| Sans `uplift_promo` | 0,282 | 0,781 | structurante → gardée |
| Sans `ratio_samedi_hp` | 0,287 | 0,631 | structurante → gardée |
| Sans les 3 variables de promo | 0,356 | 0,057 | l'axe promo **est** la segmentation |
| Avec `Promo2` | 0,250 | 0,983 | absorbée par le comportement |
| Avec `StoreType` one-hot | 0,294 | 0,101 | reproduit l'échec de la v1 (cluster de 17) |

**Deux enseignements** :

1. **La corrélation ne prouve pas la redondance.** `ratio_samedi_hp` corrèle à
   −0,68 avec l'élasticité, mais son retrait fait tomber l'ARI à 0,631 : elle
   porte une information propre. Seul le test par retrait permet de trancher.
2. **`Promo2` est absorbée** (ARI 0,983) : la structure comportementale contient
   déjà l'information administrative. En v2, la même variable *créait* le
   découpage. La différence entre les deux situations est la preuve que la
   segmentation finale est bien comportementale.

**Configuration finale retenue (5 variables)** :
`log_ca_median`, `coef_variation`, `ouvert_dimanche`, `uplift_promo`,
`ratio_samedi_hp`. `log_competition` reste utilisée comme variable
**descriptive** des segments, pas comme variable de calcul.

---

## D9 — Nombre de clusters : k = 3

| Argument | Mesure |
|---|---|
| Coude | baisse d'inertie qui décroche après k = 3 |
| Silhouette | **maximum à k = 3 (0,360)**, chute à 0,277 dès k = 4 |
| Lisibilité métier | 3 segments nommables, aucun cluster fourre-tout |

Segments obtenus (configuration finale à 5 variables) :
Standard / régulier 683 magasins · Sensible aux promotions 399 · Proximité 33.

---

## D10 — Validation finale

**Stabilité** : 30 rééchantillonnages à 80 % des magasins,
**ARI moyen 0,973, médian 0,983, minimum 0,900** sur la configuration finale à
5 variables. Aucun tirage ne s'effondre : le segment proximité (33 magasins,
3 % de l'effectif) réapparaît dans tous les rééchantillonnages. La stabilité
était moins bonne avec 6 variables (minimum 0,451), ce qui confirme le bénéfice
du retrait de `log_competition`.

**Validations externes** (variables jamais fournies à l'algorithme) :

| Segment | StoreType | Trafic | Panier |
|---|---|---|---|
| Standard / régulier | a 43 %, d 40 % | 780 clients/j | 9,85 € |
| Sensible aux promotions | **a 74,5 %** | 658 clients/j | 9,25 € |
| Proximité | **b 51,5 %** (17/17 des type b) | **1 388 clients/j** | **6,90 €** |

**Promo2** est désormais équilibrée entre les segments (0,512 / 0,532 / 0,273) :
le biais administratif a disparu.

---

## Limites assumées

- Segmentation **descriptive** : les segments ne sont pas une vérité latente,
  ils résument des comportements observés sur 2013–2015.
- **Pas de validation temporelle** du clustering : les données de test n'ont pas
  de `Customers`. Les segments sont stables à l'échantillonnage, pas prouvés
  stables dans le temps — piste d'amélioration identifiée.
- `CompetitionDistance` manquante pour 3 magasins, imputée par la médiane de
  leur type de magasin.
- Le segment proximité ne compte que 33 magasins : ses statistiques sont
  fragiles et doivent être lues comme indicatives.
