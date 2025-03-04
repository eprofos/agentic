#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'embedding de texte pour indexation vectorielle.

Ce script parcourt récursivement tous les fichiers dans un répertoire spécifié,
extrait leur contenu textuel, génère des embeddings vectoriels à l'aide du modèle
nomic-embed-text d'Ollama, et stocke ces vecteurs dans une base de données PostgreSQL
avec l'extension pgvector.

Caractéristiques:
- Parcours récursif de répertoires
- Traitement de divers types de fichiers
- Génération d'embeddings via Ollama
- Stockage dans PostgreSQL avec pgvector
- Gestion des erreurs et reprise après interruption
- Logging détaillé et suivi de progression
- Validation des embeddings stockés
"""

import os
import sys
import time
import hashlib
import mimetypes
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any, Union

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from langchain_ollama import OllamaEmbeddings

# Chargement des variables d'environnement
# Chercher le fichier .env à la racine du projet
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    "embed_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Définition des modèles SQLAlchemy
Base = declarative_base()

class Document(Base):
    """Modèle pour stocker les documents et leurs métadonnées."""
    
    __tablename__ = "documents"
    
    id = sa.Column(sa.Integer, primary_key=True)
    file_path = sa.Column(sa.String, nullable=False, unique=True)
    file_hash = sa.Column(sa.String, nullable=False)
    content_type = sa.Column(sa.String, nullable=True)
    meta_data = sa.Column(JSONB, nullable=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_path='{self.file_path}')>"


class Embedding(Base):
    """Modèle pour stocker les embeddings vectoriels."""
    
    __tablename__ = "embeddings"
    
    id = sa.Column(sa.Integer, primary_key=True)
    document_id = sa.Column(sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = sa.Column(sa.Integer, nullable=False)
    chunk_text = sa.Column(sa.Text, nullable=False)
    embedding = sa.Column(Vector(os.getenv("EMBEDDING_DIMENSION", 768)), nullable=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        sa.UniqueConstraint('document_id', 'chunk_index', name='uix_document_chunk'),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class ProcessingStatus(Base):
    """Modèle pour suivre l'état de traitement des fichiers."""
    
    __tablename__ = "processing_status"
    
    id = sa.Column(sa.Integer, primary_key=True)
    file_path = sa.Column(sa.String, nullable=False, unique=True)
    status = sa.Column(sa.String, nullable=False)  # 'pending', 'processing', 'completed', 'failed'
    error_message = sa.Column(sa.Text, nullable=True)
    attempts = sa.Column(sa.Integer, default=0)
    last_attempt = sa.Column(sa.DateTime, nullable=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessingStatus(id={self.id}, file_path='{self.file_path}', status='{self.status}')>"


class FileProcessor:
    """Classe pour traiter les fichiers et extraire leur contenu textuel."""
    
    # Extensions de fichiers textuels courants
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml',
        '.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bash',
        '.c', '.cpp', '.h', '.hpp', '.java', '.kt', '.rs', '.go', '.rb',
        '.php', '.pl', '.pm', '.sql', '.r', '.swift', '.m', '.mm'
    }
    
    # Extensions de fichiers à ignorer
    IGNORE_EXTENSIONS = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.dat',
        '.db', '.sqlite', '.sqlite3', 
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg', '.webp',
        # Vidéo
        '.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv',
        # Audio
        '.mp3', '.wav', '.ogg', '.flac',
        # Fichiers compressés
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z', '.tar.gz',
        # Fichiers de configuration
        '.env', '.gitignore', '.dockerignore', 
        # Fichiers de verrouillage
        '.lock', 'package-lock.json', 'yarn.lock',
        # Code compilé
        '.class',
        # Données binaires
        '.parquet', '.avro',
        # Logs
        '.log'
    }
    
    # Répertoires à ignorer
    IGNORE_DIRS = {
        '__pycache__', '.git', '.svn', '.hg', '.idea', '.vscode',
        'node_modules', 'venv', 'env', '.env', '.venv', 'dist', 'build'
    }
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        """
        Initialise le processeur de fichiers.
        
        Args:
            max_file_size: Taille maximale des fichiers à traiter (en octets)
        """
        self.max_file_size = max_file_size
    
    def should_process_file(self, file_path: Path) -> bool:
        """
        Détermine si un fichier doit être traité.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            bool: True si le fichier doit être traité, False sinon
        """
        # Vérifier si le fichier existe
        if not file_path.exists() or not file_path.is_file():
            return False
        
        # Vérifier la taille du fichier
        if file_path.stat().st_size > self.max_file_size:
            logger.warning(f"Fichier trop volumineux: {file_path} ({file_path.stat().st_size} octets)")
            return False
        
        # Vérifier l'extension du fichier
        extension = file_path.suffix.lower()
        if extension in self.IGNORE_EXTENSIONS:
            return False
        
        # Vérifier le type MIME du fichier
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and not mime_type.startswith(('text/', 'application/json', 'application/xml')):
            return False
        
        return True
    
    def should_process_dir(self, dir_path: Path) -> bool:
        """
        Détermine si un répertoire doit être traité.
        
        Args:
            dir_path: Chemin du répertoire
            
        Returns:
            bool: True si le répertoire doit être traité, False sinon
        """
        # Vérifier si le répertoire existe
        if not dir_path.exists() or not dir_path.is_dir():
            return False
        
        # Vérifier si le répertoire est à ignorer
        if dir_path.name in self.IGNORE_DIRS:
            return False
        
        return True
    
    def extract_text_content(self, file_path: Path) -> str:
        """
        Extrait le contenu textuel d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            str: Contenu textuel du fichier
            
        Raises:
            UnicodeDecodeError: Si le fichier ne peut pas être décodé en UTF-8
            IOError: Si le fichier ne peut pas être lu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            logger.warning(f"Impossible de décoder le fichier en UTF-8: {file_path}")
            # Essayer avec une autre encodage
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            raise
    
    def get_file_hash(self, file_path: Path) -> str:
        """
        Calcule le hash SHA-256 d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            str: Hash SHA-256 du fichier
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Récupère les métadonnées d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Dict[str, Any]: Métadonnées du fichier
        """
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix,
            "mime_type": mimetypes.guess_type(str(file_path))[0],
            "relative_path": str(file_path),
        }


class TextChunker:
    """Classe pour découper le texte en chunks pour l'embedding."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le découpeur de texte.
        
        Args:
            chunk_size: Taille maximale des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks (en caractères)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Découpe le texte en chunks.
        
        Args:
            text: Texte à découper
            
        Returns:
            List[str]: Liste des chunks
        """
        if not text:
            return []
        
        # Découper le texte en paragraphes
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph)
            
            # Si le paragraphe est plus grand que la taille maximale, le découper
            if paragraph_size > self.chunk_size:
                # Ajouter le chunk courant s'il existe
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Découper le paragraphe en phrases
                sentences = self._split_into_sentences(paragraph)
                
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if temp_size + sentence_size <= self.chunk_size:
                        temp_chunk.append(sentence)
                        temp_size += sentence_size
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_size = sentence_size
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # Si le paragraphe peut être ajouté au chunk courant
            elif current_size + paragraph_size <= self.chunk_size:
                current_chunk.append(paragraph)
                current_size += paragraph_size
            
            # Sinon, créer un nouveau chunk
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
        
        # Ajouter le dernier chunk s'il existe
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Ajouter des chunks avec chevauchement si nécessaire
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            
            for i in range(len(chunks)):
                overlapped_chunks.append(chunks[i])
                
                if i < len(chunks) - 1:
                    # Créer un chunk de chevauchement
                    current_chunk = chunks[i]
                    next_chunk = chunks[i + 1]
                    
                    # Prendre la fin du chunk courant
                    current_end = current_chunk[-(self.chunk_overlap // 2):]
                    
                    # Prendre le début du chunk suivant
                    next_start = next_chunk[:self.chunk_overlap // 2]
                    
                    # Créer le chunk de chevauchement
                    overlap_chunk = current_end + next_start
                    
                    # Ajouter le chunk de chevauchement s'il n'est pas trop petit
                    if len(overlap_chunk) >= self.chunk_size // 4:
                        overlapped_chunks.append(overlap_chunk)
            
            return overlapped_chunks
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Découpe un texte en phrases.
        
        Args:
            text: Texte à découper
            
        Returns:
            List[str]: Liste des phrases
        """
        # Règles simples pour découper en phrases
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class EmbeddingGenerator:
    """Classe pour générer des embeddings à partir de texte."""
    
    def __init__(self, model_name: str = "nomic-embed-text", batch_size: int = 10):
        """
        Initialise le générateur d'embeddings.
        
        Args:
            model_name: Nom du modèle d'embedding
            batch_size: Nombre de textes à traiter par lot
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialiser le modèle d'embedding
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=f"{ollama_host}:{ollama_port}"
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Liste des textes
            
        Returns:
            List[List[float]]: Liste des embeddings
        """
        if not texts:
            return []
        
        # Traiter les textes par lots
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Erreur lors de la génération des embeddings: {e}")
                # En cas d'erreur, réessayer avec un seul texte à la fois
                for text in batch:
                    try:
                        embedding = self.embeddings.embed_documents([text])
                        all_embeddings.extend(embedding)
                    except Exception as e:
                        logger.error(f"Erreur lors de la génération de l'embedding pour le texte: {e}")
                        # Ajouter un embedding vide
                        all_embeddings.append([0.0] * int(os.getenv("EMBEDDING_DIMENSION", 768)))
        
        return all_embeddings


class DatabaseManager:
    """Classe pour gérer la connexion à la base de données et les opérations CRUD."""
    
    def __init__(self):
        """Initialise le gestionnaire de base de données."""
        # Récupérer les informations de connexion depuis les variables d'environnement
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "vector_db")
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        
        # Créer l'URL de connexion
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Créer le moteur SQLAlchemy
        self.engine = sa.create_engine(db_url)
        
        # Créer les tables si elles n'existent pas
        Base.metadata.create_all(self.engine)
        
        # Créer une session
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """
        Récupère une session de base de données.
        
        Returns:
            Session: Session SQLAlchemy
        """
        return self.Session()
    
    def create_extension(self):
        """Crée l'extension pgvector si elle n'existe pas."""
        with self.engine.connect() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    
    def clear_database(self):
        """Nettoie la base de données en supprimant toutes les données."""
        logger.info("Nettoyage de la base de données...")
        with self.engine.connect() as conn:
            conn.execute(sa.text("TRUNCATE TABLE embeddings CASCADE"))
            conn.execute(sa.text("TRUNCATE TABLE documents CASCADE"))
            conn.execute(sa.text("TRUNCATE TABLE processing_status CASCADE"))
            conn.commit()
        logger.info("Base de données nettoyée avec succès")
    
    def get_processed_files(self) -> Set[str]:
        """
        Récupère la liste des fichiers déjà traités.
        
        Returns:
            Set[str]: Ensemble des chemins de fichiers traités
        """
        with self.get_session() as session:
            processed_files = session.query(Document.file_path).all()
            return {file[0] for file in processed_files}
    
    def get_failed_files(self) -> Set[str]:
        """
        Récupère la liste des fichiers dont le traitement a échoué.
        
        Returns:
            Set[str]: Ensemble des chemins de fichiers en échec
        """
        with self.get_session() as session:
            failed_files = session.query(ProcessingStatus.file_path).filter(
                ProcessingStatus.status == 'failed'
            ).all()
            return {file[0] for file in failed_files}
    
    def save_document(self, file_path: str, file_hash: str, content_type: str, metadata: Dict[str, Any]) -> int:
        """
        Enregistre un document dans la base de données.
        
        Args:
            file_path: Chemin du fichier
            file_hash: Hash du fichier
            content_type: Type de contenu du fichier
            metadata: Métadonnées du fichier
            
        Returns:
            int: ID du document
        """
        with self.get_session() as session:
            # Vérifier si le document existe déjà
            existing_doc = session.query(Document).filter(Document.file_path == file_path).first()
            
            if existing_doc:
                # Mettre à jour le document existant
                existing_doc.file_hash = file_hash
                existing_doc.content_type = content_type
                existing_doc.meta_data = metadata
                existing_doc.updated_at = datetime.utcnow()
                session.commit()
                return existing_doc.id
            
            # Créer un nouveau document
            document = Document(
                file_path=file_path,
                file_hash=file_hash,
                content_type=content_type,
                meta_data=metadata
            )
            session.add(document)
            session.commit()
            return document.id
    
    def save_embeddings(self, document_id: int, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Enregistre les embeddings dans la base de données.
        
        Args:
            document_id: ID du document
            chunks: Liste des chunks de texte
            embeddings: Liste des embeddings
        """
        with self.get_session() as session:
            # Supprimer les embeddings existants pour ce document
            session.query(Embedding).filter(Embedding.document_id == document_id).delete()
            
            # Ajouter les nouveaux embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                embedding_obj = Embedding(
                    document_id=document_id,
                    chunk_index=i,
                    chunk_text=chunk,
                    embedding=embedding
                )
                session.add(embedding_obj)
            
            session.commit()
    
    def update_processing_status(self, file_path: str, status: str, error_message: Optional[str] = None) -> None:
        """
        Met à jour le statut de traitement d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            status: Statut de traitement ('pending', 'processing', 'completed', 'failed')
            error_message: Message d'erreur (si status == 'failed')
        """
        with self.get_session() as session:
            # Vérifier si le statut existe déjà
            existing_status = session.query(ProcessingStatus).filter(
                ProcessingStatus.file_path == file_path
            ).first()
            
            if existing_status:
                # Mettre à jour le statut existant
                existing_status.status = status
                existing_status.error_message = error_message
                existing_status.attempts += 1
                existing_status.last_attempt = datetime.utcnow()
                existing_status.updated_at = datetime.utcnow()
            else:
                # Créer un nouveau statut
                processing_status = ProcessingStatus(
                    file_path=file_path,
                    status=status,
                    error_message=error_message,
                    attempts=1,
                    last_attempt=datetime.utcnow()
                )
                session.add(processing_status)
            
            session.commit()
    
    def validate_embeddings(self, document_id: int) -> bool:
        """
        Valide que les embeddings ont été correctement stockés.
        
        Args:
            document_id: ID du document
            
        Returns:
            bool: True si les embeddings sont valides, False sinon
        """
        with self.get_session() as session:
            # Vérifier que les embeddings existent
            count = session.query(sa.func.count(Embedding.id)).filter(
                Embedding.document_id == document_id
            ).scalar()
            
            return count > 0


class EmbeddingPipeline:
    """Classe principale pour orchestrer le processus d'embedding."""
    
    def __init__(
        self,
        source_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 10,
        max_retries: int = 3
    ):
        """
        Initialise le pipeline d'embedding.
        
        Args:
            source_dir: Répertoire source contenant les fichiers à traiter
            chunk_size: Taille maximale des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks (en caractères)
            batch_size: Nombre de textes à traiter par lot
            max_retries: Nombre maximal de tentatives en cas d'échec
        """
        self.source_dir = Path(source_dir).resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialiser les composants
        self.file_processor = FileProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(
            model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            batch_size=batch_size
        )
        self.db_manager = DatabaseManager()
        
        # Créer l'extension pgvector
        self.db_manager.create_extension()
        
        # Récupérer les fichiers déjà traités
        self.processed_files = self.db_manager.get_processed_files()
        self.failed_files = self.db_manager.get_failed_files()
        
        logger.info(f"Initialisation du pipeline d'embedding pour le répertoire: {self.source_dir}")
        logger.info(f"Nombre de fichiers déjà traités: {len(self.processed_files)}")
        logger.info(f"Nombre de fichiers en échec: {len(self.failed_files)}")
    
    def discover_files(self) -> List[Path]:
        """
        Découvre tous les fichiers à traiter dans le répertoire source.
        
        Returns:
            List[Path]: Liste des chemins de fichiers à traiter
        """
        all_files = []
        
        for root, dirs, files in os.walk(self.source_dir):
            # Filtrer les répertoires à ignorer
            dirs[:] = [d for d in dirs if self.file_processor.should_process_dir(Path(root) / d)]
            
            for file in files:
                file_path = Path(root) / file
                
                # Vérifier si le fichier doit être traité
                if self.file_processor.should_process_file(file_path):
                    all_files.append(file_path)
        
        return all_files
    
    def process_file(self, file_path: Path) -> bool:
        """
        Traite un fichier et génère des embeddings.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            bool: True si le traitement a réussi, False sinon
        """
        rel_path = str(file_path.relative_to(self.source_dir))
        
        # Vérifier si le fichier a déjà été traité
        if rel_path in self.processed_files:
            logger.debug(f"Fichier déjà traité: {rel_path}")
            return True
        
        logger.info(f"Traitement du fichier: {rel_path}")
        
        try:
            # Mettre à jour le statut de traitement
            self.db_manager.update_processing_status(rel_path, 'processing')
            
            # Extraire le contenu textuel
            content = self.file_processor.extract_text_content(file_path)
            
            # Calculer le hash du fichier
            file_hash = self.file_processor.get_file_hash(file_path)
            
            # Récupérer les métadonnées du fichier
            metadata = self.file_processor.get_file_metadata(file_path)
            
            # Récupérer le type MIME du fichier
            content_type, _ = mimetypes.guess_type(str(file_path))
            
            # Enregistrer le document
            document_id = self.db_manager.save_document(
                rel_path, file_hash, content_type, metadata
            )
            
            # Découper le texte en chunks
            chunks = self.text_chunker.split_text(content)
            
            if not chunks:
                logger.warning(f"Aucun chunk généré pour le fichier: {rel_path}")
                self.db_manager.update_processing_status(rel_path, 'completed')
                return True
            
            # Générer les embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Enregistrer les embeddings
            self.db_manager.save_embeddings(document_id, chunks, embeddings)
            
            # Valider les embeddings
            if not self.db_manager.validate_embeddings(document_id):
                raise ValueError("Les embeddings n'ont pas été correctement stockés")
            
            # Mettre à jour le statut de traitement
            self.db_manager.update_processing_status(rel_path, 'completed')
            
            # Ajouter le fichier à la liste des fichiers traités
            self.processed_files.add(rel_path)
            
            logger.info(f"Fichier traité avec succès: {rel_path} ({len(chunks)} chunks)")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {rel_path}: {e}")
            self.db_manager.update_processing_status(rel_path, 'failed', str(e))
            return False
    
    def run(self) -> Tuple[int, int, int]:
        """
        Exécute le pipeline d'embedding.
        
        Returns:
            Tuple[int, int, int]: (nombre de fichiers traités, nombre de fichiers en échec, nombre de fichiers ignorés)
        """
        logger.info("Démarrage du pipeline d'embedding")
        
        # Découvrir les fichiers à traiter
        all_files = self.discover_files()
        logger.info(f"Nombre total de fichiers découverts: {len(all_files)}")
        
        # Filtrer les fichiers déjà traités
        files_to_process = [
            f for f in all_files
            if str(f.relative_to(self.source_dir)) not in self.processed_files
        ]
        logger.info(f"Nombre de fichiers à traiter: {len(files_to_process)}")
        
        # Traiter les fichiers
        success_count = 0
        failure_count = 0
        ignored_count = len(all_files) - len(files_to_process)
        
        with tqdm(total=len(files_to_process), desc="Traitement des fichiers") as pbar:
            for file_path in files_to_process:
                rel_path = str(file_path.relative_to(self.source_dir))
                
                # Vérifier si le fichier a déjà échoué trop de fois
                if rel_path in self.failed_files:
                    with self.db_manager.get_session() as session:
                        status = session.query(ProcessingStatus).filter(
                            ProcessingStatus.file_path == rel_path
                        ).first()
                        
                        if status and status.attempts >= self.max_retries:
                            logger.warning(f"Fichier ignoré (trop d'échecs): {rel_path}")
                            pbar.update(1)
                            ignored_count += 1
                            continue
                
                # Traiter le fichier
                success = self.process_file(file_path)
                
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                
                pbar.update(1)
        
        logger.info(f"Pipeline d'embedding terminé")
        logger.info(f"Nombre de fichiers traités avec succès: {success_count}")
        logger.info(f"Nombre de fichiers en échec: {failure_count}")
        logger.info(f"Nombre de fichiers ignorés: {ignored_count}")
        
        return success_count, failure_count, ignored_count


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Génère des embeddings pour des fichiers textuels.")
    parser.add_argument("--clear-db", action="store_true", help="Nettoie la base de données avant de commencer")
    parser.add_argument("--source-dir", type=str, default="../libs", help="Répertoire source contenant les fichiers à traiter")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Taille maximale des chunks (en caractères)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chevauchement entre les chunks (en caractères)")
    parser.add_argument("--batch-size", type=int, default=10, help="Nombre de textes à traiter par lot")
    parser.add_argument("--max-retries", type=int, default=3, help="Nombre maximal de tentatives en cas d'échec")
    args = parser.parse_args()

    # Initialiser le gestionnaire de base de données
    db_manager = DatabaseManager()
    
    # Nettoyer la base de données si demandé
    if args.clear_db:
        db_manager.clear_database()
    
    # Créer et exécuter le pipeline
    pipeline = EmbeddingPipeline(
        source_dir=args.source_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    success_count, failure_count, ignored_count = pipeline.run()
    
    # Afficher un résumé
    print("\nRésumé:")
    print(f"Nombre de fichiers traités avec succès: {success_count}")
    print(f"Nombre de fichiers en échec: {failure_count}")
    print(f"Nombre de fichiers ignorés: {ignored_count}")
    
    # Retourner un code d'erreur si des fichiers ont échoué
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()