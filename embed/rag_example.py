#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des embeddings pour un système RAG.

Ce script montre comment utiliser les embeddings générés et stockés dans
la base de données PostgreSQL pour créer un système de Retrieval-Augmented
Generation (RAG) qui peut répondre à des questions basées sur les documents
indexés.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import cast
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

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
    logs_dir / "rag_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


class RAGSystem:
    """Système de Retrieval-Augmented Generation (RAG)."""
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:1b",
        top_k: int = 5
    ):
        """
        Initialise le système RAG.
        
        Args:
            embedding_model: Nom du modèle d'embedding
            llm_model: Nom du modèle de langage
            top_k: Nombre de documents à récupérer
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        
        # Récupérer les informations de connexion depuis les variables d'environnement
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "vector_db")
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        
        # Créer l'URL de connexion
        self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Créer le moteur SQLAlchemy
        self.engine = sa.create_engine(self.db_url)
        
        # Créer une session
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialiser le modèle d'embedding
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=f"{ollama_host}:{ollama_port}"
        )
        
        # Initialiser le modèle de langage
        self.llm = Ollama(
            model=llm_model,
            base_url=f"{ollama_host}:{ollama_port}"
        )
        
        # Définir le template de prompt pour le RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Tu es un assistant IA qui répond à des questions en utilisant uniquement les informations fournies dans le contexte ci-dessous.
Si tu ne connais pas la réponse ou si l'information n'est pas présente dans le contexte, indique-le clairement.
Ne fabrique pas d'informations et n'utilise pas de connaissances externes.

Contexte:
{context}

Question: {question}

Réponse:"""
        )
        
        # Créer la chaîne LLM
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        logger.info(f"Système RAG initialisé avec le modèle d'embedding {embedding_model} et le modèle de langage {llm_model}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Génère l'embedding pour une requête.
        
        Args:
            query: Requête à encoder
            
        Returns:
            List[float]: Embedding de la requête
        """
        return self.embeddings.embed_query(query)
    
    def retrieve_relevant_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Récupère les chunks les plus pertinents pour une requête.
        
        Args:
            query_embedding: Embedding de la requête
            
        Returns:
            List[Dict[str, Any]]: Liste des chunks pertinents avec leurs métadonnées
        """
        with self.Session() as session:
            # Requête SQL pour récupérer les chunks les plus similaires
            query = sa.text(f"""
                SELECT 
                    e.chunk_text,
                    e.document_id,
                    d.file_path,
                    1 - (e.embedding <=> cast(:query_embedding AS vector)) AS similarity
                FROM 
                    embeddings e
                JOIN 
                    documents d ON e.document_id = d.id
                ORDER BY 
                    e.embedding <=> cast(:query_embedding AS vector)
                LIMIT :top_k
            """)
            
            result = session.execute(
                query,
                {
                    "query_embedding": query_embedding,
                    "top_k": self.top_k
                }
            )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_text": row.chunk_text,
                    "document_id": row.document_id,
                    "file_path": row.file_path,
                    "similarity": row.similarity
                })
            
            return chunks
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formate les chunks en un contexte pour le LLM.
        
        Args:
            chunks: Liste des chunks pertinents
            
        Returns:
            str: Contexte formaté
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Document {i+1}: {chunk['file_path']}]\n{chunk['chunk_text']}")
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str, verbose: bool = False) -> str:
        """
        Répond à une question en utilisant le système RAG.
        
        Args:
            question: Question à répondre
            verbose: Afficher les détails du processus
            
        Returns:
            str: Réponse à la question
        """
        logger.info(f"Question: {question}")
        
        # Générer l'embedding de la question
        query_embedding = self.generate_query_embedding(question)
        
        # Récupérer les chunks pertinents
        chunks = self.retrieve_relevant_chunks(query_embedding)
        
        if verbose:
            print("\n=== Chunks pertinents ===")
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i+1} (Similarité: {chunk['similarity']:.4f}) ---")
                print(f"Source: {chunk['file_path']}")
                print(f"Texte: {chunk['chunk_text'][:200]}...")
        
        # Formater le contexte
        context = self.format_context(chunks)
        
        # Générer la réponse
        result = self.chain.invoke({"context": context, "question": question})
        response = result.get("text", "")  # Extraire la réponse du résultat
        
        logger.info(f"Réponse générée avec {len(chunks)} chunks pertinents")
        
        return response


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Système RAG pour répondre à des questions basées sur les documents indexés.")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="Nom du modèle d'embedding")
    parser.add_argument("--llm-model", type=str, default="llama3.2:1b", help="Nom du modèle de langage")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de documents à récupérer")
    parser.add_argument("--verbose", action="store_true", help="Afficher les détails du processus")
    args = parser.parse_args()
    
    # Créer le système RAG
    rag_system = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    print("\n=== Système RAG ===")
    print(f"Modèle d'embedding: {args.embedding_model}")
    print(f"Modèle de langage: {args.llm_model}")
    print(f"Nombre de documents à récupérer: {args.top_k}")
    print("\nTapez 'exit' pour quitter.")
    
    # Boucle interactive
    while True:
        print("\n")
        question = input("Question: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        if not question.strip():
            continue
        
        try:
            answer = rag_system.answer_question(question, verbose=args.verbose)
            print("\nRéponse:")
            print(answer)
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            print(f"\nErreur: {e}")


if __name__ == "__main__":
    main()