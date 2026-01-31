# 🚀 Guide Complet Polars LazyFrame

**Guide de référence pour travailler avec Polars LazyFrame en Data Science**

---

## 📋 Table des matières

1. [Création et Chargement](#1-création-et-chargement)
2. [Informations sur le LazyFrame](#2-informations-sur-le-lazyframe)
3. [Sélection de Colonnes](#3-sélection-de-colonnes)
4. [Min, Max et Statistiques Descriptives](#4-min-max-et-statistiques-descriptives)
5. [Filtrage (Filter)](#5-filtrage-filter)
6. [Tri (Sort)](#6-tri-sort)
7. [Ajout et Modification de Colonnes](#7-ajout-et-modification-de-colonnes)
8. [GroupBy (Agrégations)](#8-groupby-agrégations)
9. [Jointures](#9-jointures)
10. [Autres Commandes Utiles](#10-autres-commandes-utiles)
11. [Expressions Avancées](#11-expressions-avancées)
12. [Optimisation et Performance](#12-optimisation-et-performance)

---

## 1. Création et Chargement

### Créer un LazyFrame

```python
import polars as pl

# Depuis un CSV (lazy)
lf = pl.scan_csv("data.csv")

# Depuis un Parquet (lazy)
lf = pl.scan_parquet("data.parquet")

# Depuis un DataFrame
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
lf = df.lazy()

# Depuis plusieurs fichiers
lf = pl.scan_csv("data/*.csv")  # Tous les CSV d'un dossier
```

### Exécuter le LazyFrame

```python
# Collecter toutes les données
df = lf.collect()

# Collecter avec streaming (pour gros fichiers)
df = lf.collect(streaming=True)

# Collecter seulement N lignes
df = lf.fetch(n_rows=1000)
```

---

## 2. Informations sur le LazyFrame

```python
# Schéma (types des colonnes)
lf.collect_schema()
lf.schema
lf.dtypes

# Noms des colonnes
lf.columns

# Nombre de lignes et colonnes (nécessite collect)
lf.collect().shape              # (n_lignes, n_colonnes)
lf.collect().height             # Nombre de lignes
lf.collect().width              # Nombre de colonnes

# Aperçu des données
lf.head(10).collect()           # 10 premières lignes
lf.tail(10).collect()           # 10 dernières lignes
lf.limit(100).collect()         # Limiter à 100 lignes

# Afficher le plan d'exécution (pour optimisation)
print(lf.explain())             # Plan logique
print(lf.explain(optimized=True))  # Plan optimisé
```

---

## 3. Sélection de Colonnes

```python
# Sélectionner des colonnes spécifiques
lf.select("col1", "col2")
lf.select(pl.col("prix"), pl.col("quantite"))

# Sélectionner toutes les colonnes
lf.select(pl.all())

# Sélectionner avec regex
lf.select(pl.col("^ventes.*$"))  # Colonnes commençant par "ventes"

# Exclure des colonnes
lf.select(pl.exclude("id", "date"))
lf.select(pl.all().exclude("id"))

# Sélectionner par type
lf.select(pl.col(pl.Float64))   # Toutes colonnes Float64
lf.select(pl.col(pl.NUMERIC_DTYPES))  # Toutes colonnes numériques
lf.select(pl.col(pl.String))    # Toutes colonnes String

# Renommer
lf.rename({"ancien_nom": "nouveau_nom"})
lf.select(pl.col("prix").alias("price"))
```

---

## 4. Min, Max et Statistiques Descriptives

### Min et Max

```python
# Min/Max d'une colonne
lf.select(pl.col("prix").min()).collect()
lf.select(pl.col("prix").max()).collect()

# Min/Max de plusieurs colonnes
lf.select([
    pl.col("prix").min().alias("prix_min"),
    pl.col("prix").max().alias("prix_max")
]).collect()

# Min/Max par groupe
lf.group_by("categorie").agg([
    pl.col("prix").min(),
    pl.col("prix").max()
]).collect()
```

### Statistiques Descriptives

```python
# Stats complètes (nécessite DataFrame)
lf.collect().describe()

# Stats individuelles
lf.select([
    pl.col("prix").mean().alias("moyenne"),
    pl.col("prix").median().alias("mediane"),
    pl.col("prix").std().alias("ecart_type"),
    pl.col("prix").var().alias("variance"),
    pl.col("prix").sum().alias("total"),
    pl.col("prix").count().alias("nombre"),
    pl.col("prix").n_unique().alias("valeurs_uniques"),
    pl.col("prix").quantile(0.25).alias("Q1"),
    pl.col("prix").quantile(0.75).alias("Q3")
]).collect()

# Statistiques conditionnelles
lf.select([
    pl.col("prix").filter(pl.col("prix") > 100).mean().alias("prix_moyen_sup_100"),
    pl.col("prix").filter(pl.col("categorie") == "A").sum().alias("total_cat_A")
]).collect()
```

---

## 5. Filtrage (Filter)

### Filtres de base

```python
# Égalité
lf.filter(pl.col("prix") == 100)
lf.filter(pl.col("ville") == "Paris")

# Comparaisons
lf.filter(pl.col("prix") > 100)
lf.filter(pl.col("prix") >= 100)
lf.filter(pl.col("prix") < 50)
lf.filter(pl.col("prix") <= 50)
lf.filter(pl.col("prix") != 0)

# Between
lf.filter(pl.col("prix").is_between(10, 100))
lf.filter((pl.col("prix") > 10) & (pl.col("prix") < 100))
```

### Filtres multiples

```python
# AND (ET)
lf.filter(
    (pl.col("prix") > 100) & 
    (pl.col("quantite") < 50)
)

# OR (OU)
lf.filter(
    (pl.col("ville") == "Paris") | 
    (pl.col("ville") == "Lyon")
)

# Combinaison AND et OR
lf.filter(
    ((pl.col("prix") > 100) & (pl.col("quantite") > 10)) |
    (pl.col("categorie") == "Premium")
)
```

### Filtres sur listes

```python
# In (appartient à)
lf.filter(pl.col("ville").is_in(["Paris", "Lyon", "Marseille"]))

# Not in
lf.filter(~pl.col("ville").is_in(["Paris", "Lyon"]))
```

### Filtres sur valeurs nulles

```python
# Null
lf.filter(pl.col("email").is_null())

# Not null
lf.filter(pl.col("email").is_not_null())

# Remplacer les nulls avant filtrage
lf.filter(pl.col("prix").fill_null(0) > 0)
```

### Filtres sur strings

```python
# Contient
lf.filter(pl.col("nom").str.contains("dupont"))
lf.filter(pl.col("nom").str.contains("(?i)dupont"))  # Case insensitive

# Commence par
lf.filter(pl.col("code").str.starts_with("FR"))

# Finit par
lf.filter(pl.col("email").str.ends_with("@gmail.com"))

# Longueur
lf.filter(pl.col("nom").str.len_chars() > 5)

# Match regex
lf.filter(pl.col("code").str.contains(r"^[A-Z]{2}\d{4}$"))
```

### Filtres sur dates

```python
# Comparaison de dates
lf.filter(pl.col("date") >= "2024-01-01")
lf.filter(pl.col("date").is_between("2024-01-01", "2024-12-31"))

# Année
lf.filter(pl.col("date").dt.year() == 2024)

# Mois
lf.filter(pl.col("date").dt.month() == 1)

# Jour de la semaine
lf.filter(pl.col("date").dt.weekday() == 0)  # Lundi
```

---

## 6. Tri (Sort)

```python
# Tri ascendant
lf.sort("prix")
lf.sort(pl.col("prix"))

# Tri descendant
lf.sort("prix", descending=True)

# Tri par plusieurs colonnes
lf.sort(["ville", "prix"])
lf.sort(["ville", "prix"], descending=[False, True])

# Tri avec nulls
lf.sort("prix", nulls_last=True)   # Nulls à la fin
lf.sort("prix", nulls_last=False)  # Nulls au début

# Tri par expression
lf.sort(pl.col("prix") * pl.col("quantite"))
```

---

## 7. Ajout et Modification de Colonnes

### Ajouter des colonnes

```python
# Ajouter une colonne
lf.with_columns(
    (pl.col("prix") * pl.col("quantite")).alias("total")
)

# Ajouter plusieurs colonnes
lf.with_columns([
    (pl.col("prix") * 1.2).alias("prix_ttc"),
    (pl.col("prix") * 0.2).alias("tva"),
    pl.col("nom").str.to_uppercase().alias("nom_majuscule")
])

# Colonne avec valeur constante
lf.with_columns(
    pl.lit(100).alias("constante"),
    pl.lit("France").alias("pays")
)
```

### Modifier des colonnes existantes

```python
# Remplacer une colonne
lf.with_columns(
    pl.col("prix").round(2).alias("prix")
)

# Modifier le type
lf.with_columns(
    pl.col("prix").cast(pl.Int64).alias("prix")
)

# Transformation conditionnelle
lf.with_columns(
    pl.when(pl.col("prix") > 100)
      .then(pl.lit("Cher"))
      .otherwise(pl.lit("Pas cher"))
      .alias("categorie_prix")
)
```

### Operations sur colonnes

```python
# Mathématiques
lf.with_columns([
    (pl.col("a") + pl.col("b")).alias("somme"),
    (pl.col("a") - pl.col("b")).alias("difference"),
    (pl.col("a") * pl.col("b")).alias("produit"),
    (pl.col("a") / pl.col("b")).alias("division"),
    (pl.col("a") % pl.col("b")).alias("modulo"),
    (pl.col("a") ** 2).alias("carre")
])

# Strings
lf.with_columns([
    pl.col("nom").str.to_uppercase().alias("nom_maj"),
    pl.col("nom").str.to_lowercase().alias("nom_min"),
    pl.col("nom").str.strip_chars().alias("nom_nettoye"),
    pl.col("prenom").str.concat(pl.col("nom"), separator=" ").alias("nom_complet"),
    pl.col("texte").str.replace("a", "A").alias("texte_modifie"),
    pl.col("code").str.slice(0, 3).alias("code_court")
])

# Dates
lf.with_columns([
    pl.col("date").dt.year().alias("annee"),
    pl.col("date").dt.month().alias("mois"),
    pl.col("date").dt.day().alias("jour"),
    pl.col("date").dt.weekday().alias("jour_semaine"),
    pl.col("date").dt.date().alias("date_seule")
])
```

---

## 8. GroupBy (Agrégations)

### GroupBy simple

```python
# Grouper et agréger
lf.group_by("ville").agg([
    pl.col("ventes").sum().alias("total_ventes"),
    pl.col("ventes").mean().alias("moyenne_ventes"),
    pl.col("ventes").max().alias("max_ventes"),
    pl.col("ventes").min().alias("min_ventes"),
    pl.len().alias("nb_lignes")
])

# Grouper par plusieurs colonnes
lf.group_by(["ville", "categorie"]).agg(
    pl.col("prix").mean()
)
```

### Agrégations avancées

```python
# Multiple agrégations sur une colonne
lf.group_by("ville").agg([
    pl.col("prix").min(),
    pl.col("prix").max(),
    pl.col("prix").mean(),
    pl.col("prix").median(),
    pl.col("prix").std()
])

# Agrégations sur plusieurs colonnes
lf.group_by("categorie").agg([
    pl.col("prix").sum(),
    pl.col("quantite").sum(),
    pl.col("client_id").n_unique(),
    pl.len()
])

# First et Last
lf.group_by("ville").agg([
    pl.col("date").first().alias("premiere_vente"),
    pl.col("date").last().alias("derniere_vente")
])

# Compter les valeurs uniques
lf.group_by("ville").agg([
    pl.col("client_id").n_unique().alias("nb_clients_uniques"),
    pl.col("produit_id").n_unique().alias("nb_produits_differents")
])
```

### Filtrer après GroupBy

```python
# Having (filtrer les groupes)
lf.group_by("ville").agg([
    pl.col("ventes").sum().alias("total")
]).filter(pl.col("total") > 10000)
```

---

## 9. Jointures

### Types de jointures

```python
lf1 = pl.scan_csv("ventes.csv")
lf2 = pl.scan_csv("clients.csv")

# Inner join
lf1.join(lf2, on="client_id", how="inner")

# Left join
lf1.join(lf2, on="client_id", how="left")

# Right join (équivalent à left avec tables inversées)
lf1.join(lf2, on="client_id", how="right")

# Outer join (full)
lf1.join(lf2, on="client_id", how="outer")

# Cross join (produit cartésien)
lf1.join(lf2, how="cross")

# Anti join (lignes de lf1 sans correspondance dans lf2)
lf1.join(lf2, on="client_id", how="anti")

# Semi join (lignes de lf1 avec correspondance dans lf2)
lf1.join(lf2, on="client_id", how="semi")
```

### Jointures avec colonnes différentes

```python
# Colonnes différentes
lf1.join(lf2, left_on="id_client", right_on="id", how="left")

# Plusieurs colonnes
lf1.join(lf2, on=["client_id", "date"], how="inner")

# Suffixes pour colonnes en double
lf1.join(lf2, on="client_id", how="left", suffix="_client")
```

---

## 10. Autres Commandes Utiles

### Dédoublonnage

```python
# Supprimer les doublons
lf.unique()

# Supprimer doublons sur colonnes spécifiques
lf.unique(subset=["client_id"])
lf.unique(subset=["ville", "date"])

# Garder première ou dernière occurrence
lf.unique(subset=["client_id"], keep="first")
lf.unique(subset=["client_id"], keep="last")
```

### Compter

```python
# Nombre total de lignes
lf.select(pl.len()).collect()

# Nombre de lignes par groupe
lf.group_by("ville").agg(pl.len())

# Valeurs uniques
lf.select(pl.col("ville").n_unique()).collect()
lf.select(pl.col("ville").unique()).collect()
```

### Valeurs manquantes

```python
# Compter les nulls
lf.null_count().collect()

# Nulls par colonne
lf.select(pl.col("email").is_null().sum()).collect()

# Remplacer les nulls
lf.with_columns(
    pl.col("prix").fill_null(0)
)

# Remplacer avec stratégie
lf.with_columns(
    pl.col("prix").fill_null(strategy="forward")  # Propagation avant
)

# Supprimer lignes avec nulls
lf.drop_nulls()
lf.drop_nulls(subset=["email", "telephone"])
```

### Échantillonnage

```python
# Échantillon aléatoire (nécessite collect)
lf.collect().sample(n=1000)           # 1000 lignes
lf.collect().sample(fraction=0.1)     # 10% des données
lf.collect().sample(n=1000, seed=42)  # Reproductible
```

### Limiter et découper

```python
# Limiter le nombre de lignes
lf.limit(100)

# Découper (slice)
lf.slice(offset=10, length=100)  # 100 lignes à partir de la 10ème

# Head et tail
lf.head(10)
lf.tail(10)
```

### Transposition et reshape

```python
# Pivot (nécessite collect)
lf.collect().pivot(
    values="montant",
    index="date",
    columns="categorie"
)

# Unpivot (melt)
lf.melt(
    id_vars=["date"],
    value_vars=["prix1", "prix2", "prix3"]
)
```

### Concaténation

```python
# Concaténer verticalement (empiler)
pl.concat([lf1, lf2], how="vertical")

# Concaténer horizontalement (côte à côte)
pl.concat([lf1, lf2], how="horizontal")

# Concaténer avec alignement
pl.concat([lf1, lf2], how="diagonal")
```

---

## 11. Expressions Avancées

### When-Then-Otherwise

```python
# Conditions simples
lf.with_columns(
    pl.when(pl.col("prix") > 100)
      .then(pl.lit("Cher"))
      .otherwise(pl.lit("Pas cher"))
      .alias("categorie")
)

# Conditions multiples (if-elif-else)
lf.with_columns(
    pl.when(pl.col("prix") < 50)
      .then(pl.lit("Bon marché"))
      .when(pl.col("prix") < 100)
      .then(pl.lit("Moyen"))
      .otherwise(pl.lit("Cher"))
      .alias("categorie")
)
```

### Expressions sur plusieurs colonnes

```python
# Opérations sur toutes les colonnes
lf.select(pl.all().sum())
lf.select(pl.all().mean())

# Sélectionner et transformer
lf.select([
    pl.col(pl.Float64).round(2),
    pl.col(pl.String).str.to_uppercase()
])

# Appliquer une fonction à toutes colonnes numériques
lf.with_columns([
    (pl.col(col) * 1.2).alias(f"{col}_augmente") 
    for col in lf.columns 
    if lf.schema[col] in [pl.Float64, pl.Int64]
])
```

### Fenêtres (Window Functions)

```python
# Rang
lf.with_columns(
    pl.col("prix").rank().over("categorie").alias("rang")
)

# Row number
lf.with_columns(
    pl.col("prix").cum_count().over("ville").alias("numero_ligne")
)

# Moyenne mobile
lf.with_columns(
    pl.col("ventes").rolling_mean(window_size=7).alias("moyenne_7j")
)

# Cumulative
lf.with_columns([
    pl.col("ventes").cum_sum().alias("cumul_ventes"),
    pl.col("ventes").cum_max().alias("max_cumul"),
    pl.col("ventes").cum_min().alias("min_cumul")
])

# Lag et Lead
lf.with_columns([
    pl.col("prix").shift(1).alias("prix_precedent"),
    pl.col("prix").shift(-1).alias("prix_suivant")
])
```

---

## 12. Optimisation et Performance

### Streaming

```python
# Pour fichiers très volumineux
lf.collect(streaming=True)
```

### Projection Pushdown

```python
# ✅ Bon : Sélection AVANT le filtre
lf.select(["col1", "col2"]).filter(pl.col("col1") > 10)

# ❌ Moins optimal
lf.filter(pl.col("col1") > 10).select(["col1", "col2"])
```

### Predicate Pushdown

```python
# Polars optimise automatiquement en appliquant les filtres tôt
lf.filter(pl.col("prix") > 100).group_by("ville").agg(pl.col("ventes").sum())
# Le filtre est appliqué AVANT le groupby automatiquement
```

### Parallélisation

```python
# Polars parallélise automatiquement
# Mais on peut contrôler avec :
import os
os.environ["POLARS_MAX_THREADS"] = "8"
```

### Cache intermédiaire

```python
# Pour réutiliser un LazyFrame
lf_filtered = lf.filter(pl.col("prix") > 100).cache()

# Utiliser plusieurs fois
result1 = lf_filtered.group_by("ville").agg(pl.col("ventes").sum())
result2 = lf_filtered.group_by("categorie").agg(pl.col("prix").mean())
```

---

## 📊 Exemples Complets

### Pipeline complet d'analyse

```python
import polars as pl

result = (
    pl.scan_csv("ventes.csv")
    # Nettoyage
    .filter(pl.col("montant").is_not_null())
    .filter(pl.col("montant") > 0)
    .filter(pl.col("date") >= "2024-01-01")
    
    # Transformation
    .with_columns([
        (pl.col("prix") * pl.col("quantite")).alias("total"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date_parsed"),
        pl.col("ville").str.to_uppercase().alias("ville_upper")
    ])
    
    # Agrégation
    .group_by(["ville_upper", "categorie"])
    .agg([
        pl.col("total").sum().alias("CA_total"),
        pl.col("total").mean().alias("CA_moyen"),
        pl.col("client_id").n_unique().alias("nb_clients"),
        pl.len().alias("nb_ventes")
    ])
    
    # Filtrage post-agrégation
    .filter(pl.col("CA_total") > 10000)
    
    # Tri
    .sort("CA_total", descending=True)
    
    # Exécution
    .collect()
)
```

### Join et agrégation

```python
ventes = pl.scan_csv("ventes.csv")
clients = pl.scan_csv("clients.csv")
produits = pl.scan_csv("produits.csv")

result = (
    ventes
    .join(clients, on="client_id", how="left")
    .join(produits, on="produit_id", how="left")
    .with_columns([
        (pl.col("prix_unitaire") * pl.col("quantite")).alias("montant_total")
    ])
    .group_by(["region", "categorie_produit"])
    .agg([
        pl.col("montant_total").sum().alias("CA"),
        pl.col("commande_id").n_unique().alias("nb_commandes"),
        pl.col("client_id").n_unique().alias("nb_clients")
    ])
    .sort("CA", descending=True)
    .collect()
)
```

---

## 🎯 Bonnes Pratiques

1. **Utilisez LazyFrame** pour les gros fichiers → optimisation automatique
2. **Sélectionnez tôt** → ne chargez que les colonnes nécessaires
3. **Filtrez tôt** → réduisez le volume de données rapidement
4. **Utilisez `.collect()` à la fin** → une seule exécution
5. **Utilisez `streaming=True`** pour fichiers > RAM
6. **Chaînez les opérations** → meilleure lisibilité et optimisation
7. **Préférez Polars expressions** (`pl.col()`) aux lambda functions

---

## 📚 Ressources

- Documentation officielle : https://docs.pola.rs/
- GitHub : https://github.com/pola-rs/polars
- Comparaison Pandas : https://docs.pola.rs/user-guide/migration/pandas/

---

**Guide créé pour Data Science M2 - Polars LazyFrame**
