# Méthodologie — segmenter une population d'entités (magasins, clients, produits)

Checklist réutilisable. Contexte d'application : segmentation de 1 115 magasins
Rossmann à partir de leur historique de ventes.

---

## Phase 1 — Cadrer avant de coder

1. **Écrire la question métier** : « quels groupes de magasins se comportent
   différemment, et que ferait-on différemment pour chacun ? »
   Un segment sans action associée n'a aucune valeur.
2. **Choisir l'unité** (magasin, client, produit) et **la période d'observation**.
   La période doit être assez longue pour être représentative, assez récente
   pour être actuelle.
3. **Se demander si les segments doivent être actionnables** : un cluster défini
   par une variable non observable en production est inutilisable.

## Phase 2 — Construire la table entité × variables

4. **Une ligne par entité**, une colonne par caractéristique.
5. **Choisir le périmètre des agrégats** : moyens calculés sur les jours
   d'ouverture uniquement, jamais sur les jours fermés (sinon on mesure des
   fermetures, pas de l'activité).
6. **Préférer des mesures robustes** : médiane plutôt que moyenne si les
   valeurs extrêmes comptent.
7. **Normaliser par la taille** quand on veut mesurer une forme, pas un volume
   (coefficient de variation plutôt qu'écart-type brut).

## Phase 3 — Contrôler et nettoyer

8. **Compter les entités et les doublons** après chaque jointure
   (`validate="one_to_one"` permet de le prouver).
9. **Lister les valeurs manquantes** et choisir un traitement explicite :
   - peu de manquants (< 1 %) : imputation simple, choix documenté ;
   - beaucoup : une catégorie « inconnu » ou une variable indicatrice.
10. **Éliminer les variables quasi constantes** : `écart-type / moyenne < 5 %`
    ou un mode couvrant > 95 % des entités. Elles n'apportent pas d'information,
    mais la standardisation les rendra artificiellement importantes.
11. **Éliminer les redondances** : corrélation `|r| > 0,9` → garder une seule
    variable, la plus interprétable.

## Phase 4 — Mettre les variables sur un pied d'égalité

12. **Corriger l'asymétrie** (`log1p`) pour les variables à longue queue
    (distances, montants, populations). Vérifier que le `skew` se rapproche de 0.
13. **Encoder les catégorielles** (one-hot). Surveiller les catégories rares :
    une colonne indicatrice à 9 modalités sur 1 115 crée des points isolés.
14. **Standardiser** (`StandardScaler`) : obligatoire pour KMeans, qui repose
    sur une distance euclidienne.
    ⚠️ La standardisation est un **choix de pondération** : elle déclare que
    toutes les dimensions comptent autant. Ce n'est pas neutre.

## Phase 5 — Choisir le nombre de clusters

15. **Méthode du coude** sur l'inertie intra-classe (à lire comme un ordre de
    grandeur, pas comme un optimum mathématique).
16. **Silhouette** : < 0,25 structure faible ; 0,25–0,50 correcte ; > 0,50 nette.
    ⚠️ La silhouette favorise les clusters convexes et compacts ; elle mesure
    une géométrie, pas une utilité.
17. **Stabilité** : relancer sur des sous-échantillons et mesurer la
    concordance (indice ARI). Un découpage instable ne doit pas être publié.
18. **Interprétabilité** : un k qui produit un cluster « fourre-tout » de 600
    magasins et trois micro-clusters n'est pas exploitable.

## Phase 6 — Interpréter et valider

19. **Désinverser les transformations** pour décrire les centroïdes en unités
    réelles (euros, mètres). Un centroïde en écart-type ne se raconte pas.
20. **Nommer chaque segment** en une phrase métier.
21. **Vérifier les clusters par une variable non utilisée dans le clustering**
    (validation externe). Si les segments ne diffèrent pas sur des variables
    qu'on n'a pas données à l'algorithme, ils ne capturent rien de réel.
22. **Visualiser en 2D** par ACP, sans prétendre que les clusters y sont nets.
23. **Documenter le caractère exploratoire** : les segments sont une
    construction, pas une vérité découverte.

## Pièges classiques

- KMeans renvoie toujours `k` clusters, même sur du bruit. Sans validation,
  on publie une illusion.
- Un cluster de 3 magasins est presque toujours un artefact d'échelle ou de
  catégorie rare.
- Les distances entre clusters n'ont pas de sens absolu.
- Trop de dimensions dilue les différences : en grande dimension, toutes les
  distances se ressemblent.
- Comparer toujours le résultat à une segmentation naïve et lisible
  (par exemple des terciles de chiffre d'affaires) pour vérifier l'apport réel.
