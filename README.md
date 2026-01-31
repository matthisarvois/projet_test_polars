# Faire tourner le projet

### Si vous n'avez pas `uv` :
```
pip install uv
```

### Synchronisez l'environnement :
```
uv sync
```

### lancez le projet (à la racine) :
```
uv run streamlit run src/frontend/application.py
```

# Prochaine amélioration :
- Ajout d'alias
- Ajout d'un `__main__.py` propre
- Ajout d'analyses avec d'autres bases de données pour le module polars
- Création d'API pour fluidifier  les requêtes*
- Déploiment de l'application en plusieurs dockers