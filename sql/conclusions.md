# Analyse SQL — Ventes Rossmann

> Base : 1 017 209 lignes de ventes journalières × 1 115 magasins (2013-2015).
> Outil : DuckDB. Toutes les moyennes excluent les jours de fermeture (`Open = 1`).

## 1. Types de magasins : les totaux trompent

- Le type `a` domine en CA total (3 165 M€), mais il compte beaucoup trop de magasins
  par rapport aux autres types (602 contre 17 seulement pour le type `b`). On ne peut
  donc pas conclure que les magasins de type `a` ont un meilleur rendement.
- Les effectifs sont très déséquilibrés : 602 magasins de type `a`, 348 de type `d`,
  148 de type `c` et 17 de type `b`.
- Normalisé : le type `b` est le plus rentable (9,4 M€/magasin contre ~5,2 M€ pour
  les autres types).
- Deux des cinq meilleurs magasins sont de type `b` malgré ses 17 magasins. On peut
  donc conclure que les magasins de type `b` ont un bien meilleur rendement — résultat
  confirmé par le CA moyen par magasin, où le type `b` arrive en tête.

## 2. Promos : +38,8 % observés

- La vente moyenne journalière est supérieure de 38,8 % les jours avec promo
  (8,2 k€ contre 5,9 k€). Écart observé, pas effet prouvé : les promos tombent
  souvent sur des jours déjà porteurs.

## 3. Jours de semaine

- Après calcul des ventes moyennes par jour, on remarque que le top des ventes est
  réalisé le premier jour de la semaine, le lundi — ce qui est logique puisque les
  gens reviennent du week-end et que tous les magasins sont ouverts. On remarque
  aussi que les ventes moyennes du dimanche sont égales à celles du lundi, ce qui
  est absurde à première lecture puisque le dimanche, la quasi-totalité des
  magasins sont fermés.
- Explication : moyenne calculée sur un petit nombre de magasins atypiques, donc
  non comparable aux autres jours. La vérification par type montre que tous les
  magasins de type `b` ouvrent le dimanche (17/17), avec 11,4 k€ — encore plus que
  leur moyenne habituelle (10,2 k€).
- Donc « le dimanche vend beaucoup » = « le dimanche, seuls les meilleurs magasins
  ouvrent ». L'effet dimanche EST l'effet type-b. Les maths ferment — moyenne
  pondérée : (2216 × 11,4 + 1100 × 2,9 + 277 × 3,5) / 3593 ≈ 8,2 k€, exactement
  la moyenne observée le dimanche.

## 4. Top 5 magasins

- Dans le top 5 figurent deux magasins de type `b` et trois de type `a`, ce qui
  confirme le meilleur rendement des magasins de type `b`. L'assortiment ne semble
  pas discriminer : le top 5 mélange des magasins basiques (a) et étendus (c).

## Limites de cette analyse

- Période couverte : 2013-2015 (Allemagne uniquement) — résultats non généralisables.
- L'effet promo (+38,8 %) est un écart observé, pas une mesure causale.
- Les moyennes journalières dépendent des périmètres d'ouverture, très variables
  selon les jours (dimanche) et les types de magasins.