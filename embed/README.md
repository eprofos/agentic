# Système d'Embedding Vectoriel

Ce système permet de parcourir récursivement tous les fichiers d'un répertoire, d'extraire leur contenu textuel, de générer des embeddings vectoriels à l'aide du modèle `nomic-embed-text` d'Ollama, et de stocker ces vecteurs dans une base de données PostgreSQL avec l'extension pgvector.

## Caractéristiques

- Parcours récursif de répertoires
- Traitement de divers types de fichiers textuels
- Génération d'embeddings via Ollama
- Stockage dans PostgreSQL avec pgvector
- Gestion des erreurs et reprise après interruption
- Logging détaillé et suivi de progression
- Validation des embeddings stockés

## Prérequis

- Python 3.8+
- Docker et Docker Compose
- PostgreSQL avec l'extension pgvector
- Ollama avec le modèle nomic-embed-text

## Installation

1. Cloner le dépôt et accéder au répertoire du projet:

```bash
cd embed
```

2. Installer les dépendances:

```bash
pip install -r requirements.txt
```

3. Vérifier que le fichier `.env` à la racine du projet contient les paramètres de connexion corrects pour PostgreSQL et Ollama.

## Configuration

Le fichier `.env` à la racine du projet doit contenir les variables d'environnement suivantes:

```
# PostgreSQL Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=vector_db
POSTGRES_PORT=5432
POSTGRES_HOST=localhost

# Ollama Configuration
OLLAMA_HOST=http://localhost
OLLAMA_PORT=11434

# Embedding Configuration
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# Application Configuration
LOG_LEVEL=INFO
BATCH_SIZE=10
```

## Démarrage des services

Avant d'exécuter le script d'embedding, assurez-vous que les services PostgreSQL et Ollama sont en cours d'exécution:

```bash
docker-compose up -d
```

Assurez-vous que le modèle `nomic-embed-text` est disponible dans Ollama:

```bash
docker exec -it ollama-service ollama pull nomic-embed-text
```

## Utilisation

Pour exécuter le script d'embedding avec les paramètres par défaut:

```bash
python embed.py
```

Options disponibles:

- `--source-dir`: Répertoire source contenant les fichiers à traiter (par défaut: "../libs")
- `--chunk-size`: Taille maximale des chunks en caractères (par défaut: 1000)
- `--chunk-overlap`: Chevauchement entre les chunks en caractères (par défaut: 200)
- `--batch-size`: Nombre de textes à traiter par lot (par défaut: 10)
- `--max-retries`: Nombre maximal de tentatives en cas d'échec (par défaut: 3)

Exemple avec des options personnalisées:

```bash
python embed.py --source-dir "../libs" --chunk-size 1500 --chunk-overlap 300 --batch-size 5
```

## Structure de la base de données

Le script crée les tables suivantes dans la base de données:

1. `documents`: Stocke les métadonnées des fichiers traités
2. `embeddings`: Stocke les embeddings vectoriels générés
3. `processing_status`: Suit l'état de traitement des fichiers

## Reprise après interruption

Le script est conçu pour reprendre le traitement après une interruption. Il garde une trace des fichiers déjà traités et ne les retraite pas, sauf si le contenu du fichier a changé.

## Logging

Les logs sont écrits dans la console et dans un fichier `embed_{time}.log`. Le niveau de log peut être configuré via la variable d'environnement `LOG_LEVEL`.

## Validation

Le script valide que les embeddings ont été correctement stockés dans la base de données après chaque traitement de fichier.

## Exemple d'utilisation avec LangChain

Une fois les embeddings générés, vous pouvez les utiliser avec LangChain pour créer un système RAG (Retrieval-Augmented Generation):

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Configuration de la connexion à la base de données
CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/vectordb"
COLLECTION_NAME = "document_embeddings"

# Initialisation du modèle d'embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialisation du vectorstore
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Initialisation du modèle de langage
llm = Ollama(model="llama3")

# Création d'une chaîne de question-réponse
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Exemple d'utilisation
query = "Quelle est la fonctionnalité principale de pydantic-ai?"
result = qa_chain.run(query)
print(result)
```

## Dépannage

Si vous rencontrez des problèmes:

1. Vérifiez que les services PostgreSQL et Ollama sont en cours d'exécution
2. Vérifiez que le modèle `nomic-embed-text` est disponible dans Ollama
3. Vérifiez les logs pour identifier les erreurs
4. Assurez-vous que les variables d'environnement sont correctement configurées

## Limitations

- Le script est conçu pour traiter des fichiers textuels. Les fichiers binaires sont ignorés.
- La taille maximale des fichiers est limitée à 10 Mo par défaut.
- Le script ne traite pas les fichiers dans certains répertoires spécifiques (comme `.git`, `__pycache__`, etc.).