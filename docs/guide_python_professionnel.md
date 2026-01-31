# 🚀 Guide Complet Python Professionnel

**Guide de référence pour gérer un projet Python moderne : pyproject.toml, Docker, API et Async**

---

## 📋 Table des matières

1. [Gestion de pyproject.toml](#1-gestion-de-pyprojecttoml)
2. [Déploiement avec Docker](#2-déploiement-avec-docker)
3. [Créer une API REST](#3-créer-une-api-rest)
4. [Fonctions Asynchrones](#4-fonctions-asynchrones)
5. [CI/CD et Automatisation](#5-cicd-et-automatisation)
6. [Best Practices](#6-best-practices)

---

## 1. Gestion de pyproject.toml

### 🎯 Qu'est-ce que pyproject.toml ?

Fichier de configuration moderne Python (PEP 518) qui remplace `setup.py`, `requirements.txt`, `setup.cfg`, etc.

### 📝 Structure complète

```toml
[project]
name = "mon-projet"
version = "0.1.0"
description = "Description courte du projet"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Ton Nom", email = "ton.email@example.com"}
]
maintainers = [
    {name = "Ton Nom", email = "ton.email@example.com"}
]
keywords = ["data", "analytics", "streamlit"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

# Dépendances principales
dependencies = [
    "streamlit>=1.28.0",
    "polars>=0.19.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.4.0",
    "httpx>=0.25.0",
]

# Dépendances optionnelles (extras)
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "ruff>=0.0.280",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.2.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]

# URLs du projet
[project.urls]
Homepage = "https://github.com/username/mon-projet"
Documentation = "https://mon-projet.readthedocs.io"
Repository = "https://github.com/username/mon-projet"
Issues = "https://github.com/username/mon-projet/issues"

# Scripts CLI (optionnel)
[project.scripts]
mon-app = "src.main:main"
mon-api = "src.api.main:start_server"

# Build system
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Configuration Hatchling
[tool.hatch.build.targets.wheel]
packages = ["src"]

# Configuration Ruff (linter moderne)
[tool.ruff]
line-length = 100
target-version = "py310"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (géré par black)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Ignore unused imports in __init__

# Configuration Black (formatter)
[tool.black]
line-length = 100
target-version = ['py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

# Configuration pytest
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
]
testpaths = ["tests"]
pythonpath = ["."]

# Configuration mypy (type checking)
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

# Configuration coverage
[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "**/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### 🔧 Commandes essentielles

```bash
# Installation du projet en mode éditable
uv pip install -e .

# Avec extras dev
uv pip install -e ".[dev]"

# Avec plusieurs extras
uv pip install -e ".[dev,test,docs]"

# Ajouter une dépendance
uv add fastapi
uv add --dev pytest  # Dépendance de développement

# Retirer une dépendance
uv remove fastapi

# Synchroniser l'environnement
uv sync

# Mettre à jour les dépendances
uv lock --upgrade

# Exporter requirements.txt (pour compatibilité)
uv pip compile pyproject.toml -o requirements.txt
```

### 📦 Publier le package

```bash
# Build
python -m build

# Upload sur PyPI (test)
twine upload --repository testpypi dist/*

# Upload sur PyPI (production)
twine upload dist/*
```

---

## 2. Déploiement avec Docker

### 🐳 Dockerfile multi-stage optimisé

**Dockerfile**
```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml ./

# Créer l'environnement virtuel et installer les dépendances
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache-dir -r pyproject.toml

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Créer un utilisateur non-root
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Copier l'environnement virtuel depuis le builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Répertoire de travail
WORKDIR /app

# Copier le code source
COPY --chown=appuser:appuser . .

# Installer le projet
RUN /opt/venv/bin/pip install --no-cache-dir -e .

# Changer vers utilisateur non-root
USER appuser

# Port exposé
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Commande par défaut
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 🐳 Dockerfile pour Streamlit

**Dockerfile.streamlit**
```dockerfile
FROM python:3.11-slim

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copier et installer dépendances
COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r pyproject.toml

ENV PATH="/opt/venv/bin:$PATH"

# Copier le code
COPY . .
RUN /opt/venv/bin/pip install --no-cache-dir -e .

# Port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Lancer Streamlit
CMD ["streamlit", "run", "src/frontend/application.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 🐳 docker-compose.yml

```yaml
version: '3.8'

services:
  # API FastAPI
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mon-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data:ro
    depends_on:
      - db
      - redis
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Frontend Streamlit
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: mon-streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    networks:
      - app-network
    restart: unless-stopped

  # Base de données PostgreSQL
  db:
    image: postgres:15-alpine
    container_name: mon-postgres
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network
    restart: unless-stopped

  # Redis pour le cache
  redis:
    image: redis:7-alpine
    container_name: mon-redis
    ports:
      - "6379:6379"
    networks:
      - app-network
    restart: unless-stopped

  # Nginx reverse proxy (optionnel)
  nginx:
    image: nginx:alpine
    container_name: mon-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - streamlit
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge
```

### 🔧 Commandes Docker

```bash
# Build
docker build -t mon-app:latest .

# Run simple
docker run -p 8000:8000 mon-app:latest

# Docker Compose
docker-compose up -d                    # Lancer en arrière-plan
docker-compose up --build              # Rebuild et lancer
docker-compose down                     # Arrêter
docker-compose down -v                  # Arrêter et supprimer volumes
docker-compose logs -f api              # Voir les logs
docker-compose exec api bash            # Entrer dans le container

# Nettoyage
docker system prune -a                  # Nettoyer images inutilisées
docker volume prune                     # Nettoyer volumes inutilisés
```

### 📝 .dockerignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Tests
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/

# Data (sauf structure)
data/*.csv
!data/.gitkeep

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore
```

---

## 3. Créer une API REST

### 🚀 FastAPI - Structure complète

**src/api/main.py**
```python
"""Point d'entrée de l'API FastAPI"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from src.api.routers import data, analytics, health
from src.api.dependencies import get_db
from src.backend.database import init_db, close_db

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info("🚀 Starting API...")
    await init_db()
    yield
    # Shutdown
    logger.info("🛑 Shutting down API...")
    await close_db()


# Création de l'application
app = FastAPI(
    title="Mon API Data",
    description="API pour analyser des données avec Polars",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: liste spécifique
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API Data Analytics",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestion des erreurs HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


def start_server():
    """Fonction pour lancer le serveur (utilisée dans pyproject.toml)"""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Dev seulement
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
```

### 📁 Structure des routers

**src/api/routers/health.py**
```python
"""Router pour le health check"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
import psutil

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    cpu_percent: float
    memory_percent: float


@router.get("", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Vérification de la santé de l'API"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent
    )
```

**src/api/routers/data.py**
```python
"""Router pour la gestion des données"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import polars as pl

from src.backend.main import load_data_lazy, process_data

router = APIRouter()


class DataInfo(BaseModel):
    """Modèle pour les informations sur les données"""
    n_rows: int = Field(..., description="Nombre de lignes")
    n_cols: int = Field(..., description="Nombre de colonnes")
    columns: List[str] = Field(..., description="Liste des colonnes")
    dtypes: dict = Field(..., description="Types des colonnes")


class FilterRequest(BaseModel):
    """Modèle pour les requêtes de filtrage"""
    column: str
    operator: str = Field(..., pattern="^(==|!=|>|<|>=|<=)$")
    value: float | str | int


@router.get("/info", response_model=DataInfo)
async def get_data_info(filepath: str = Query(..., description="Chemin du fichier")):
    """Obtenir les informations sur un dataset"""
    try:
        lf = load_data_lazy(filepath)
        df = lf.collect()
        
        return DataInfo(
            n_rows=df.height,
            n_cols=df.width,
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in df.schema.items()}
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Fichier non trouvé: {filepath}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement: {str(e)}"
        )


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers CSV sont acceptés"
        )
    
    try:
        contents = await file.read()
        df = pl.read_csv(contents)
        
        # Sauvegarder
        output_path = f"data/{file.filename}"
        df.write_csv(output_path)
        
        return {
            "message": "Fichier uploadé avec succès",
            "filename": file.filename,
            "rows": df.height,
            "columns": df.width
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'upload: {str(e)}"
        )


@router.post("/filter")
async def filter_data(
    filepath: str = Query(...),
    filters: List[FilterRequest] = []
):
    """Filtrer les données"""
    try:
        lf = load_data_lazy(filepath)
        
        # Appliquer les filtres
        for f in filters:
            if f.operator == "==":
                lf = lf.filter(pl.col(f.column) == f.value)
            elif f.operator == ">":
                lf = lf.filter(pl.col(f.column) > f.value)
            # ... autres opérateurs
        
        df = lf.collect()
        
        return {
            "rows": df.height,
            "data": df.to_dicts()[:100]  # Limiter à 100 lignes
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du filtrage: {str(e)}"
        )
```

**src/api/routers/analytics.py**
```python
"""Router pour les analyses"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from src.backend.analytics import calculate_stats, detect_outliers
from src.backend.main import load_data_lazy

router = APIRouter()


class StatsResponse(BaseModel):
    """Modèle pour les statistiques"""
    column: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float


class OutlierResponse(BaseModel):
    """Modèle pour les outliers"""
    n_outliers: int
    percentage: float
    outlier_indices: List[int]


@router.get("/stats/{column}", response_model=StatsResponse)
async def get_column_stats(
    column: str,
    filepath: str = Query(..., description="Chemin du fichier")
):
    """Calculer les statistiques d'une colonne"""
    try:
        lf = load_data_lazy(filepath)
        df = lf.collect()
        
        if column not in df.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Colonne '{column}' non trouvée"
            )
        
        stats = df.select([
            pl.col(column).count().alias("count"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).quantile(0.25).alias("q25"),
            pl.col(column).quantile(0.50).alias("q50"),
            pl.col(column).quantile(0.75).alias("q75"),
        ]).to_dicts()[0]
        
        return StatsResponse(column=column, **stats)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du calcul: {str(e)}"
        )


@router.get("/outliers/{column}", response_model=OutlierResponse)
async def get_outliers(
    column: str,
    filepath: str = Query(...),
    threshold: float = Query(3.0, ge=1.0, le=5.0)
):
    """Détecter les outliers dans une colonne"""
    try:
        lf = load_data_lazy(filepath)
        df = lf.collect()
        
        df_outliers = detect_outliers(df, column, threshold)
        outliers = df_outliers.filter(pl.col("is_outlier"))
        
        return OutlierResponse(
            n_outliers=outliers.height,
            percentage=(outliers.height / df.height) * 100,
            outlier_indices=outliers.select(pl.col("row_nr")).to_series().to_list()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la détection: {str(e)}"
        )
```

### 🔐 Dépendances et authentification

**src/api/dependencies.py**
```python
"""Dépendances FastAPI"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"  # À mettre dans .env
ALGORITHM = "HS256"

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Vérifier le token JWT"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expiré"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide"
        )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Créer un token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

### 📞 Tester l'API

```bash
# Lancer l'API
uvicorn src.api.main:app --reload

# Tester avec curl
curl http://localhost:8000/health

# Upload fichier
curl -X POST "http://localhost:8000/api/v1/data/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/iris.csv"

# Obtenir stats
curl "http://localhost:8000/api/v1/analytics/stats/sepal_length?filepath=data/iris.csv"

# Avec httpx (Python)
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/health")
    print(response.json())
```

---

## 4. Fonctions Asynchrones

### 🔄 Bases de l'asynchrone en Python

**Concepts clés**
```python
import asyncio
from typing import List

# Fonction asynchrone
async def fonction_async():
    """Fonction qui peut être suspendue"""
    await asyncio.sleep(1)  # Suspend l'exécution
    return "Résultat"

# Appeler une fonction async
async def main():
    result = await fonction_async()
    print(result)

# Exécuter
asyncio.run(main())
```

### 📚 Exemples pratiques

**Exemple 1 : Requêtes HTTP parallèles**
```python
import asyncio
import httpx
from typing import List, Dict

async def fetch_url(client: httpx.AsyncClient, url: str) -> Dict:
    """Fetch une URL de manière asynchrone"""
    response = await client.get(url)
    return {"url": url, "status": response.status_code, "data": response.json()}


async def fetch_multiple_urls(urls: List[str]) -> List[Dict]:
    """Fetch plusieurs URLs en parallèle"""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_url(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results


# Utilisation
async def main():
    urls = [
        "https://api.github.com/users/octocat",
        "https://api.github.com/users/torvalds",
        "https://jsonplaceholder.typicode.com/posts/1"
    ]
    results = await fetch_multiple_urls(urls)
    for result in results:
        print(f"{result['url']}: {result['status']}")

asyncio.run(main())
```

**Exemple 2 : Traitement de données asynchrone**
```python
import asyncio
import polars as pl
from typing import List
from pathlib import Path

async def process_file_async(filepath: Path) -> pl.DataFrame:
    """Traiter un fichier de manière asynchrone"""
    # Simuler I/O
    await asyncio.sleep(0.1)
    
    # Charger et traiter
    df = pl.read_csv(filepath)
    df = df.filter(pl.col("value") > 100)
    
    return df


async def process_multiple_files(filepaths: List[Path]) -> List[pl.DataFrame]:
    """Traiter plusieurs fichiers en parallèle"""
    tasks = [process_file_async(fp) for fp in filepaths]
    results = await asyncio.gather(*tasks)
    return results


# Utilisation
async def main():
    files = [Path(f"data/file_{i}.csv") for i in range(10)]
    dataframes = await process_multiple_files(files)
    
    # Combiner les résultats
    combined = pl.concat(dataframes)
    print(f"Total rows: {combined.height}")

asyncio.run(main())
```

**Exemple 3 : API asynchrone avec FastAPI**
```python
from fastapi import FastAPI, BackgroundTasks
import asyncio
import httpx

app = FastAPI()


async def long_task(item_id: int):
    """Tâche longue exécutée en arrière-plan"""
    await asyncio.sleep(5)
    print(f"Task {item_id} completed")


@app.post("/items/{item_id}")
async def create_item(item_id: int, background_tasks: BackgroundTasks):
    """Endpoint qui lance une tâche en arrière-plan"""
    background_tasks.add_task(long_task, item_id)
    return {"message": "Item created, processing in background"}


@app.get("/external-api")
async def call_external_api():
    """Appeler une API externe de manière asynchrone"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

**Exemple 4 : Contexte manager asynchrone**
```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_database_connection():
    """Context manager asynchrone pour gérer une connexion DB"""
    # Setup
    print("Opening database connection")
    connection = await asyncio.sleep(0.1)  # Simuler connexion
    
    try:
        yield connection
    finally:
        # Teardown
        print("Closing database connection")
        await asyncio.sleep(0.1)  # Simuler fermeture


async def main():
    async with async_database_connection() as conn:
        print("Using database connection")
        # Faire des requêtes...

asyncio.run(main())
```

**Exemple 5 : Queue asynchrone (Producer/Consumer)**
```python
import asyncio
from asyncio import Queue
import random

async def producer(queue: Queue, producer_id: int):
    """Producteur qui ajoute des items dans la queue"""
    for i in range(5):
        item = f"Item-{producer_id}-{i}"
        await queue.put(item)
        print(f"Producer {producer_id} produced {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))
    await queue.put(None)  # Signal de fin


async def consumer(queue: Queue, consumer_id: int):
    """Consommateur qui traite les items de la queue"""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        
        print(f"Consumer {consumer_id} processing {item}")
        await asyncio.sleep(random.uniform(0.1, 0.3))
        queue.task_done()


async def main():
    queue = Queue(maxsize=10)
    
    # Créer producers et consumers
    producers = [asyncio.create_task(producer(queue, i)) for i in range(2)]
    consumers = [asyncio.create_task(consumer(queue, i)) for i in range(3)]
    
    # Attendre que tous les producers aient fini
    await asyncio.gather(*producers)
    
    # Attendre que la queue soit vide
    await queue.join()
    
    # Terminer les consumers
    for _ in consumers:
        await queue.put(None)
    await asyncio.gather(*consumers)

asyncio.run(main())
```

### ⚡ Optimisations async

**Avec timeout**
```python
import asyncio

async def operation_with_timeout():
    try:
        async with asyncio.timeout(5.0):  # Python 3.11+
            result = await slow_operation()
            return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None

# Python < 3.11
async def operation_with_timeout_legacy():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=5.0)
        return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
```

**Rate limiting**
```python
import asyncio
from asyncio import Semaphore

async def rate_limited_request(url: str, semaphore: Semaphore):
    """Requête avec limitation du nombre simultané"""
    async with semaphore:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()


async def main():
    # Maximum 5 requêtes simultanées
    semaphore = Semaphore(5)
    
    urls = [f"https://api.example.com/item/{i}" for i in range(100)]
    tasks = [rate_limited_request(url, semaphore) for url in urls]
    results = await asyncio.gather(*tasks)
```

**Retry avec backoff exponentiel**
```python
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs
) -> T:
    """Retry une fonction async avec backoff exponentiel"""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)


# Utilisation
async def unreliable_api_call():
    # Simuler une API instable
    if random.random() < 0.7:
        raise Exception("API error")
    return "Success"


async def main():
    result = await retry_with_backoff(unreliable_api_call)
    print(result)
```

### 🔧 Debugging async

```python
import asyncio
import logging

# Activer le debug mode
logging.basicConfig(level=logging.DEBUG)

async def buggy_function():
    await asyncio.sleep(1)
    raise ValueError("Something went wrong")


async def main():
    try:
        await buggy_function()
    except Exception as e:
        logging.exception("Error occurred")


# Exécuter avec debug
asyncio.run(main(), debug=True)
```

---

## 5. CI/CD et Automatisation

### 🔄 GitHub Actions

**.github/workflows/ci.yml**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.11"

jobs:
  # Tests et linting
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v2
      with:
        enable-cache: true
    
    - name: Set up Python
      run: uv python install ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        uv sync --all-extras
    
    - name: Lint with ruff
      run: |
        uv run ruff check .
    
    - name: Format check with black
      run: |
        uv run black --check .
    
    - name: Type check with mypy
      run: |
        uv run mypy src/
    
    - name: Run tests
      run: |
        uv run pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  # Build Docker image
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          username/mon-app:latest
          username/mon-app:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Deploy
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # SSH vers le serveur et déployer
        echo "Deploying..."
```

### 🔨 Pre-commit hooks

**.pre-commit-config.yaml**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Installation**
```bash
# Installer pre-commit
uv add --dev pre-commit

# Installer les hooks
uv run pre-commit install

# Exécuter manuellement
uv run pre-commit run --all-files
```

### 📦 Makefile

```makefile
.PHONY: install test lint format clean docker-build docker-run

# Variables
PYTHON := uv run python
PYTEST := uv run pytest
BLACK := uv run black
RUFF := uv run ruff
MYPY := uv run mypy

# Installation
install:
	uv sync --all-extras
	uv run pre-commit install

# Tests
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=html

test-fast:
	$(PYTEST) tests/ -v -x --ff

# Linting et formatting
lint:
	$(RUFF) check .
	$(MYPY) src/

format:
	$(BLACK) .
	$(RUFF) check --fix .

# Nettoyage
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov coverage.xml

# Docker
docker-build:
	docker build -t mon-app:latest .

docker-run:
	docker run -p 8000:8000 mon-app:latest

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down -v

# Développement
dev-api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

dev-streamlit:
	$(PYTHON) -m streamlit run src/frontend/application.py

# Base de données (si applicable)
db-migrate:
	$(PYTHON) -m alembic upgrade head

db-rollback:
	$(PYTHON) -m alembic downgrade -1

# Help
help:
	@echo "Commandes disponibles:"
	@echo "  make install           - Installer les dépendances"
	@echo "  make test              - Lancer les tests"
	@echo "  make lint              - Vérifier le code"
	@echo "  make format            - Formater le code"
	@echo "  make clean             - Nettoyer les fichiers temporaires"
	@echo "  make docker-build      - Build l'image Docker"
	@echo "  make docker-run        - Lancer le container Docker"
	@echo "  make dev-api           - Lancer l'API en mode dev"
	@echo "  make dev-streamlit     - Lancer Streamlit en mode dev"
```

---

## 6. Best Practices

### ✅ Structure de projet recommandée

```
mon-projet/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── data.py
│   │       └── analytics.py
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── analytics.py
│   │   ├── database.py
│   │   └── utils.py
│   └── frontend/
│       ├── __init__.py
│       ├── application.py
│       └── pages/
│           ├── __init__.py
│           └── page_1.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   └── test_backend/
├── data/
│   └── .gitkeep
├── docs/
│   └── README.md
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

### 📝 Variables d'environnement

**.env.example**
```bash
# Application
APP_NAME=mon-app
APP_ENV=development
LOG_LEVEL=info

# API
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379

# External APIs
EXTERNAL_API_KEY=your-api-key
```

**Chargement des variables**
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Mon App"
    app_env: str = "development"
    log_level: str = "info"
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: str
    
    database_url: str
    redis_url: str
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Utilisation
settings = get_settings()
print(settings.database_url)
```

### 🔒 Sécurité

```python
# 1. Ne jamais commit les secrets
# .gitignore
.env
*.key
*.pem

# 2. Utiliser des variables d'environnement
import os
SECRET_KEY = os.getenv("SECRET_KEY")

# 3. Valider les inputs
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    email: str
    age: int
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

# 4. Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/data")
@limiter.limit("5/minute")
async def get_data():
    return {"data": "..."}
```

### 📊 Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'app.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Utilisation
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

### 🧪 Tests

**tests/conftest.py**
```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_data():
    return {"key": "value"}
```

**tests/test_api/test_health.py**
```python
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## 🎓 Ressources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://docs.docker.com/
- **Asyncio**: https://docs.python.org/3/library/asyncio.html
- **uv**: https://github.com/astral-sh/uv
- **Pydantic**: https://docs.pydantic.dev/

---

**Guide créé pour le développement Python professionnel**
