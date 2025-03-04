#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour le système d'embedding.

Ce script permet de tester le système d'embedding en effectuant un test
sur un petit ensemble de données et en vérifiant que les embeddings sont
correctement générés et stockés dans la base de données.
"""

import os
import sys
import tempfile
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

# Importer les classes du script principal
from embed import (
    FileProcessor,
    TextChunker,
    EmbeddingGenerator,
    DatabaseManager,
    EmbeddingPipeline
)

# Chargement des variables d'environnement
# Chercher le fichier .env à la racine du projet
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Création du répertoire de logs s'il n'existe pas
logs_dir = Path(__file__).parent.parent / 'logs' / 'embed'
logs_dir.mkdir(parents=True, exist_ok=True)

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    logs_dir / "test_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


def create_test_files(test_dir: Path) -> None:
    """
    Crée des fichiers de test dans le répertoire spécifié.
    
    Args:
        test_dir: Répertoire où créer les fichiers de test
    """
    # Créer un fichier Python
    python_file = test_dir / "test_file.py"
    with open(python_file, "w") as f:
        f.write("""
# Exemple de fichier Python pour tester l'embedding
def hello_world():
    \"\"\"Fonction qui affiche Hello World.\"\"\"
    print("Hello, World!")

class TestClass:
    \"\"\"Classe de test.\"\"\"
    
    def __init__(self, name):
        \"\"\"Initialise la classe avec un nom.\"\"\"
        self.name = name
    
    def greet(self):
        \"\"\"Affiche un message de salutation.\"\"\"
        print(f"Hello, {self.name}!")

if __name__ == "__main__":
    hello_world()
    test = TestClass("Test")
    test.greet()
""")

    # Créer un fichier Markdown
    md_file = test_dir / "test_file.md"
    with open(md_file, "w") as f:
        f.write("""
# Titre du document

## Introduction

Ceci est un document de test pour le système d'embedding.

## Contenu principal

Le système d'embedding permet de:
- Parcourir récursivement les fichiers
- Extraire leur contenu textuel
- Générer des embeddings vectoriels
- Stocker ces vecteurs dans une base de données

## Conclusion

Ce document est utilisé pour tester le bon fonctionnement du système.
""")

    # Créer un fichier JSON
    json_file = test_dir / "test_file.json"
    with open(json_file, "w") as f:
        f.write("""
{
    "name": "Test Document",
    "type": "JSON",
    "properties": {
        "version": 1.0,
        "author": "Test Author",
        "created_at": "2023-01-01T00:00:00Z"
    },
    "data": [
        {"id": 1, "value": "First item"},
        {"id": 2, "value": "Second item"},
        {"id": 3, "value": "Third item"}
    ]
}
""")

    # Créer un sous-répertoire avec un fichier
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    
    subdir_file = subdir / "subdir_file.txt"
    with open(subdir_file, "w") as f:
        f.write("""
Ceci est un fichier texte dans un sous-répertoire.

Il permet de tester la fonctionnalité de parcours récursif du système d'embedding.

Le système devrait être capable de trouver ce fichier et de générer des embeddings pour son contenu.
""")

    print(f"Fichiers de test créés dans {test_dir}")


def test_file_processor():
    """Teste la classe FileProcessor."""
    print("\n=== Test de FileProcessor ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_test_files(temp_path)
        
        processor = FileProcessor()
        
        # Tester should_process_file
        python_file = temp_path / "test_file.py"
        assert processor.should_process_file(python_file), "Le fichier Python devrait être traité"
        
        # Tester extract_text_content
        content = processor.extract_text_content(python_file)
        assert "Hello, World!" in content, "Le contenu du fichier Python n'a pas été correctement extrait"
        
        # Tester get_file_hash
        file_hash = processor.get_file_hash(python_file)
        assert len(file_hash) == 64, "Le hash SHA-256 devrait avoir 64 caractères"
        
        # Tester get_file_metadata
        metadata = processor.get_file_metadata(python_file)
        assert metadata["extension"] == ".py", "L'extension du fichier devrait être .py"
        
        print("✅ Tous les tests de FileProcessor ont réussi")


def test_text_chunker():
    """Teste la classe TextChunker."""
    print("\n=== Test de TextChunker ===")
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # Tester split_text avec un texte court
    short_text = "Ceci est un texte court qui ne devrait pas être découpé."
    chunks = chunker.split_text(short_text)
    assert len(chunks) == 1, f"Le texte court devrait donner 1 chunk, mais a donné {len(chunks)}"
    
    # Tester split_text avec un texte long
    long_text = "\n\n".join(["Paragraphe " + str(i) + ": " + "x" * 80 for i in range(10)])
    chunks = chunker.split_text(long_text)
    assert len(chunks) > 1, f"Le texte long devrait donner plusieurs chunks, mais a donné {len(chunks)}"
    
    print("✅ Tous les tests de TextChunker ont réussi")


def test_embedding_generator():
    """Teste la classe EmbeddingGenerator."""
    print("\n=== Test de EmbeddingGenerator ===")
    
    try:
        generator = EmbeddingGenerator(batch_size=2)
        
        # Tester generate_embeddings
        texts = ["Ceci est un texte de test.", "Voici un autre texte de test."]
        embeddings = generator.generate_embeddings(texts)
        
        assert len(embeddings) == 2, f"Il devrait y avoir 2 embeddings, mais il y en a {len(embeddings)}"
        assert len(embeddings[0]) == int(os.getenv("EMBEDDING_DIMENSION", 768)), f"La dimension de l'embedding devrait être {os.getenv('EMBEDDING_DIMENSION', 768)}"
        
        print("✅ Tous les tests de EmbeddingGenerator ont réussi")
    except Exception as e:
        print(f"❌ Test de EmbeddingGenerator échoué: {e}")
        print("⚠️ Assurez-vous que le service Ollama est en cours d'exécution et que le modèle nomic-embed-text est disponible")


def test_database_manager():
    """Teste la classe DatabaseManager."""
    print("\n=== Test de DatabaseManager ===")
    
    try:
        db_manager = DatabaseManager()
        
        # Tester la connexion à la base de données
        with db_manager.get_session() as session:
            assert session is not None, "La session ne devrait pas être None"
        
        # Tester la création de l'extension pgvector
        db_manager.create_extension()
        
        print("✅ Tous les tests de DatabaseManager ont réussi")
    except Exception as e:
        print(f"❌ Test de DatabaseManager échoué: {e}")
        print("⚠️ Assurez-vous que le service PostgreSQL est en cours d'exécution et que les variables d'environnement sont correctement configurées")


def test_embedding_pipeline():
    """Teste la classe EmbeddingPipeline."""
    print("\n=== Test de EmbeddingPipeline ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_test_files(temp_path)
            
            pipeline = EmbeddingPipeline(
                source_dir=temp_dir,
                chunk_size=100,
                chunk_overlap=20,
                batch_size=2
            )
            
            # Tester discover_files
            files = pipeline.discover_files()
            assert len(files) >= 4, f"Il devrait y avoir au moins 4 fichiers, mais il y en a {len(files)}"
            
            # Tester process_file
            success = pipeline.process_file(files[0])
            assert success, f"Le traitement du fichier {files[0]} a échoué"
            
            # Tester run
            success_count, failure_count, ignored_count = pipeline.run()
            assert success_count > 0, "Au moins un fichier devrait être traité avec succès"
            
            print("✅ Tous les tests de EmbeddingPipeline ont réussi")
    except Exception as e:
        print(f"❌ Test de EmbeddingPipeline échoué: {e}")
        print("⚠️ Assurez-vous que les services PostgreSQL et Ollama sont en cours d'exécution et que les variables d'environnement sont correctement configurées")


def main():
    """Fonction principale."""
    print("=== Tests du système d'embedding ===\n")
    
    # Tester chaque composant
    test_file_processor()
    test_text_chunker()
    test_embedding_generator()
    test_database_manager()
    test_embedding_pipeline()
    
    print("\n=== Tous les tests sont terminés ===")


if __name__ == "__main__":
    main()