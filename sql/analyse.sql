--viionner les données
SELECT s.StoreType, COUNT(DISTINCT t.Store) AS nb_magasins
FROM train t
JOIN store s ON t.Store = s.Store
GROUP BY s.StoreType;

-- Q1 : chiffre d'affaires total par type de magasin
SELECT s.StoreType,
       ROUND(SUM(t.Sales) / 1000000, 1) AS ca_millions
FROM train t
JOIN store s ON t.Store = s.Store
WHERE t.Open = 1
GROUP BY s.StoreType
ORDER BY ca_millions DESC;

-- Q1-bis : CA MOYEN par magasin selon le type
SELECT s.StoreType,
       ROUND((SUM(t.Sales) / COUNT(DISTINCT t.Store)) / 1000000, 1) AS ca_millions_par_magasin
FROM train t
JOIN store s ON t.Store = s.Store
WHERE t.Open = 1
GROUP BY s.StoreType
ORDER BY ca_millions_par_magasin DESC;

--Q2 : les promos marchent-elles ? Vente moyenne avec vs sans promo
SELECT 
    CASE 
        WHEN t.Promo = 1 THEN 'Avec Promo'
        ELSE 'Sans Promo'
    END AS promo_status,
    ROUND(AVG(t.Sales) / 1000, 1) AS vente_moyenne_milliers
FROM train t
WHERE t.Open = 1
GROUP BY promo_status;

-- Q3-finale : jour qui vend le plus 
SELECT dayname(t.Date) AS jour,
       COUNT(*) AS nb_lignes,
       ROUND(AVG(t.Sales) / 1000, 1) AS vente_moyenne_milliers
FROM train t
WHERE t.Open = 1
GROUP BY jour
ORDER BY vente_moyenne_milliers DESC;

--Q3-bis : type de magasin le jour de fermeture (dimanche) vend le plus
SELECT s.StoreType,
       COUNT(DISTINCT t.Store) AS nb_magasins_ouverts,
       COUNT(*) AS nb_lignes
FROM train t JOIN store s ON t.Store = s.Store
WHERE t.Open = 1 AND t.DayOfWeek = 7
GROUP BY s.StoreType;

-- Q4 : top 5 magasins par CA, avec leur StoreType et Assortment
SELECT t.Store, s.StoreType, s.Assortment,
       ROUND(SUM(t.Sales) / 1000000, 1) AS ca_total_millions
FROM train t
JOIN store s ON t.Store = s.Store
WHERE t.Open = 1
GROUP BY t.Store, s.StoreType, s.Assortment
ORDER BY ca_total_millions DESC
LIMIT 5;