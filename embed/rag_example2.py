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
import re
from typing import List, Dict, Any
from pathlib import Path
import os.path

from loguru import logger
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import cast
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
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
        llm_model: str = None,
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
        # Récupérer les informations de configuration OpenRouter
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        
        # Utiliser le modèle spécifié ou celui de la variable d'environnement
        self.llm_model = llm_model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.7-sonnet")
        logger.info(f"Utilisation du modèle LLM: {self.llm_model}")
        
        # Initialiser le LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=openrouter_api_key,
            base_url=openrouter_api_base,
            temperature=0
        )
        
        # Définir le template de prompt pour l'extraction des mots-clés
        self.keyword_extraction_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
You are an AI assistant specialized in extracting relevant search keywords from questions.

Your task is to analyze the following question and extract the most important keywords or phrases that would be useful for a search query.

If the question is in a language other than English, extract keywords in English to ensure better search results.

Question: {question}

Extract 3-5 most relevant keywords or phrases for search (in English):
"""
        )
        
        # Créer la chaîne LLM pour l'extraction des mots-clés
        self.keyword_extraction_chain = LLMChain(
            llm=self.llm,
            prompt=self.keyword_extraction_prompt
        )
        
        # Définir le template de prompt pour le RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI assistant who provides extremely detailed answers using the information provided in the context below.
The context includes both relevant chunks and the full content of files containing those chunks.
Focus exclusively on the question asked and utilize all relevant information from the context to deliver the most comprehensive response possible with maximum specificity and depth.

Provide thorough explanations, include all pertinent details, and ensure your answer is as complete as possible.

If you don't know the answer or if the information is not present in the context, don't make up an answer.
Instead, clearly indicate them at the end of your answer in a section "KEYWORDS_FOR_NEW_SEARCH: [list of keywords]".

If you have found a satisfactory answer, don't include this keywords section.

If the question is asked in a language other than English, you should still perform the search in English.

Context:
{context}

Question: {question}

Answer:
"""
        )
        
        # Créer la chaîne LLM
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        logger.info(f"Système RAG initialisé avec le modèle d'embedding {embedding_model} et le modèle de langage {self.llm_model}")
    
    def extract_keywords_from_question(self, question: str) -> List[str]:
        """
        Extrait les mots-clés pertinents d'une question pour améliorer la recherche.
        
        Args:
            question: Question à analyser
            
        Returns:
            List[str]: Liste des mots-clés extraits
        """
        try:
            logger.info(f"Extraction des mots-clés de la question: {question}")
            result = self.keyword_extraction_chain.invoke({"question": question})
            keywords_text = result.get("text", "").strip()
            
            # Nettoyer et diviser en mots-clés individuels
            # Supprimer les numéros de liste (1., 2., etc.) et les tirets
            keywords = []
            for line in keywords_text.split('\n'):
                line = line.strip()
                # Supprimer les numéros de liste et les tirets au début
                line = re.sub(r'^(\d+\.|\-|\*)\s*', '', line)
                if line:
                    keywords.append(line)
            
            logger.info(f"Mots-clés extraits: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des mots-clés: {e}")
            # En cas d'erreur, retourner la question originale comme mot-clé
            return [question]
    
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
    
    def read_full_file_content(self, file_path: str) -> str:
        """
        Lit le contenu complet d'un fichier à partir de son chemin.
        
        Args:
            file_path: Chemin du fichier à lire
            
        Returns:
            str: Contenu complet du fichier
        """
        try:
            # Si le chemin est relatif, on considère qu'il est relatif au répertoire 'libs'
            if not os.path.isabs(file_path):
                base_path = Path(__file__).parent.parent / 'libs'
                full_path = base_path / file_path
            else:
                full_path = Path(file_path)
            
            logger.info(f"Lecture du fichier complet: {full_path}")
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            return f"Erreur lors de la lecture du fichier: {e}"
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formate les chunks en un contexte pour le LLM.
        
        Args:
            chunks: Liste des chunks pertinents
            
        Returns:
            str: Contexte formaté
        """
        context_parts = []
        
        # Garder une trace des fichiers déjà inclus pour éviter les doublons
        included_files = set()
        
        for i, chunk in enumerate(chunks):
            file_path = chunk['file_path']
            
            # Ajouter d'abord le chunk pertinent
            context_parts.append(f"[Extrait {i+1} du document: {file_path}]\n{chunk['chunk_text']}")
            
            # Si le fichier complet n'a pas encore été inclus, l'ajouter
            if file_path not in included_files:
                try:
                    full_content = self.read_full_file_content(file_path)
                    context_parts.append(f"\n[Contenu complet du fichier: {file_path}]\n{full_content}")
                    included_files.add(file_path)
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture du fichier complet {file_path}: {e}")
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str, verbose: bool = False, max_iterations: int = 3) -> str:
        """
        Répond à une question en utilisant le système RAG.
        
        Args:
            question: Question à répondre
            verbose: Afficher les détails du processus
            
            max_iterations: Nombre maximum d'itérations de recherche
            
        Returns:
            str: Réponse à la question
        """
        logger.info(f"Question: {question}")
        
        current_question = question
        original_question = question
        iteration = 0
        all_responses = []
        search_history = [question]  # Garder une trace des requêtes de recherche
        
        # Extraire les mots-clés de la question originale
        keywords = self.extract_keywords_from_question(question)
        if keywords:
            # Utiliser les mots-clés extraits pour la première recherche
            current_question = " ".join(keywords)
            search_history[0] = f"Original: '{question}' → Keywords: '{current_question}'"
            logger.info(f"Recherche avec les mots-clés extraits: {current_question}")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Itération {iteration}/{max_iterations} - Requête: {current_question}")
            
            # Générer l'embedding de la question
            query_embedding = self.generate_query_embedding(current_question)
            
            # Récupérer les chunks pertinents
            chunks = self.retrieve_relevant_chunks(query_embedding)
            
            if verbose:
                print(f"\n=== Itération {iteration}/{max_iterations} ===")
                print(f"Requête: {current_question}")
                print("\n=== Chunks pertinents ===")
                for i, chunk in enumerate(chunks):
                    print(f"\n--- Chunk {i+1} (Similarité: {chunk['similarity']:.4f}) ---")
                    print(f"Source: {chunk['file_path']}")
                    print(f"Texte: {chunk['chunk_text'][:200]}...")
            
            # Formater le contexte
            context = self.format_context(chunks)
            
            # Générer la réponse
            result = self.chain.invoke({"context": context, "question": original_question})
            response = result.get("text", "")  # Extraire la réponse du résultat
            
            logger.info(f"Réponse générée avec {len(chunks)} chunks pertinents")
            
            # Vérifier si la réponse contient des mots-clés pour une nouvelle recherche
            new_keywords = self._extract_keywords_for_new_search(response)
            
            if not new_keywords:
                # Si pas de nouveaux mots-clés, on a une réponse satisfaisante ou on ne peut pas améliorer
                all_responses.append(f"[Itération {iteration}] {response}")
                break
            else:
                # Ajouter la réponse actuelle à l'historique
                all_responses.append(f"[Itération {iteration}] {response}")
                
                # Créer une nouvelle requête avec les mots-clés
                current_question = f"{' '.join(new_keywords)}"
                search_history.append(current_question)
                
                logger.info(f"Nouveaux mots-clés identifiés: {new_keywords}")
        
        # Si on a atteint le nombre maximum d'itérations sans réponse satisfaisante
        if iteration == max_iterations and new_keywords:
            all_responses.append(
                f"\n\nAprès {max_iterations} itérations, aucune réponse satisfaisante n'a pu être trouvée. "
                f"Les recherches ont été effectuées avec les requêtes suivantes: {search_history}"
            )
        
        # Retourner toutes les réponses ou seulement la dernière selon le besoin
        if verbose:
            return "\n\n".join(all_responses)
        else:
            return all_responses[-1]
    
    def _extract_keywords_for_new_search(self, response: str) -> List[str]:
        """
        Extrait les mots-clés pour une nouvelle recherche à partir de la réponse du LLM.
        
        Args:
            response: Réponse du LLM
            
        Returns:
            List[str]: Liste des mots-clés pour une nouvelle recherche
        """
        # Chercher la section des mots-clés dans la réponse
        if "KEYWORDS_FOR_NEW_SEARCH:" in response:
            try:
                # Extraire la partie après l'indicateur
                keywords_part = response.split("KEYWORDS_FOR_NEW_SEARCH:")[1].strip()
                # Nettoyer et diviser en mots-clés individuels
                keywords = [k.strip() for k in keywords_part.split(",")]
                # Filtrer les mots-clés vides
                keywords = [k for k in keywords if k]
                return keywords
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction des mots-clés: {e}")
        
        # Si pas de mots-clés trouvés ou erreur, retourner une liste vide
        return []


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Système RAG pour répondre à des questions basées sur les documents indexés.")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="Nom du modèle d'embedding")
    parser.add_argument("--llm-model", type=str, help="Nom du modèle de langage (si non spécifié, utilise OPENROUTER_MODEL)")
    parser.add_argument("--top-k", type=int, default=10, help="Nombre de documents à récupérer")
    parser.add_argument("--max-iterations", type=int, default=3, help="Nombre maximum d'itérations de recherche")
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
    print(f"Nombre maximum d'itérations: {args.max_iterations}")
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
            answer = rag_system.answer_question(question, verbose=args.verbose, max_iterations=args.max_iterations)
            print("\nRéponse:")
            print(answer)
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            print(f"\nErreur: {e}")


if __name__ == "__main__":
    main()