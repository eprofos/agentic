#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example of using embeddings for a RAG system.

This script demonstrates how to use embeddings generated and stored in
the PostgreSQL database to create a Retrieval-Augmented Generation (RAG)
system that can answer questions based on indexed documents.
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

# Load environment variables
# Look for .env file at project root
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent / 'logs' / 'embed'
logs_dir.mkdir(parents=True, exist_ok=True)

# Logger configuration
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
    """Retrieval-Augmented Generation (RAG) System."""
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = None,
        top_k: int = 5
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the embedding model
            llm_model: Name of the language model
            top_k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        
        # Get connection information from environment variables
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_name = os.getenv("POSTGRES_DB", "vector_db")
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        
        # Create connection URL
        self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Create SQLAlchemy engine
        self.engine = sa.create_engine(self.db_url)
        
        # Create a session
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize the embedding model
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=f"{ollama_host}:{ollama_port}"
        )
        
        # Initialize the language model
        # Get OpenRouter configuration information
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        
        # Use specified model or the one from environment variable
        self.llm_model = llm_model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.7-sonnet")
        logger.info(f"Using LLM model: {self.llm_model}")
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=openrouter_api_key,
            base_url=openrouter_api_base,
            temperature=0
        )
        
        # Define the prompt template for keyword extraction
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
        
        # Create the LLM chain for keyword extraction
        self.keyword_extraction_chain = LLMChain(
            llm=self.llm,
            prompt=self.keyword_extraction_prompt
        )
        
        # Define the prompt template for RAG
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
        
        # Create the LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        logger.info(f"RAG system initialized with embedding model {embedding_model} and language model {self.llm_model}")
    
    def extract_keywords_from_question(self, question: str) -> List[str]:
        """
        Extract relevant keywords from a question to improve search.
        
        Args:
            question: Question to analyze
            
        Returns:
            List[str]: List of extracted keywords
        """
        try:
            logger.info(f"Extracting keywords from question: {question}")
            result = self.keyword_extraction_chain.invoke({"question": question})
            keywords_text = result.get("text", "").strip()
            
            # Clean and split into individual keywords
            # Remove list numbers (1., 2., etc.) and dashes
            keywords = []
            for line in keywords_text.split('\n'):
                line = line.strip()
                # Remove list numbers and dashes at the beginning
                line = re.sub(r'^(\d+\.|\-|\*)\s*', '', line)
                if line:
                    keywords.append(line)
            
            logger.info(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # In case of error, return the original question as a keyword
            return [question]
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate the embedding for a query.
        
        Args:
            query: Query to encode
            
        Returns:
            List[float]: Query embedding
        """
        return self.embeddings.embed_query(query)
    
    def retrieve_relevant_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            List[Dict[str, Any]]: List of relevant chunks with their metadata
        """
        with self.Session() as session:
            # SQL query to retrieve the most similar chunks
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
        Read the complete content of a file from its path.
        
        Args:
            file_path: Path of the file to read
            
        Returns:
            str: Complete content of the file
        """
        try:
            # If path is relative, consider it relative to 'libs' directory
            if not os.path.isabs(file_path):
                base_path = Path(__file__).parent.parent / 'libs'
                full_path = base_path / file_path
            else:
                full_path = Path(file_path)
            
            logger.info(f"Reading complete file: {full_path}")
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks into a context for the LLM.
        
        Args:
            chunks: List of relevant chunks
            
        Returns:
            str: Formatted context
        """
        context_parts = []
        
        # Keep track of already included files to avoid duplicates
        included_files = set()
        
        for i, chunk in enumerate(chunks):
            file_path = chunk['file_path']
            
            # Add the relevant chunk first
            context_parts.append(f"[Excerpt {i+1} from document: {file_path}]\n{chunk['chunk_text']}")
            
            # If the full file has not been included yet, add it
            if file_path not in included_files:
                try:
                    full_content = self.read_full_file_content(file_path)
                    context_parts.append(f"\n[Full file content: {file_path}]\n{full_content}")
                    included_files.add(file_path)
                except Exception as e:
                    logger.error(f"Error reading full file {file_path}: {e}")
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str, verbose: bool = False, max_iterations: int = 3) -> str:
        """
        Answer a question using the RAG system.
        
        Args:
            question: Question to answer
            verbose: Show process details
            
            max_iterations: Maximum number of search iterations
            
        Returns:
            str: Answer to the question
        """
        logger.info(f"Question: {question}")
        
        current_question = question
        original_question = question
        iteration = 0
        all_responses = []
        search_history = [question]  # Keep track of search queries
        
        # Extract keywords from the original question
        keywords = self.extract_keywords_from_question(question)
        if keywords:
            # Use extracted keywords for the first search
            current_question = " ".join(keywords)
            search_history[0] = f"Original: '{question}' → Keywords: '{current_question}'"
            logger.info(f"Searching with extracted keywords: {current_question}")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations} - Query: {current_question}")
            
            # Generate the embedding of the question
            query_embedding = self.generate_query_embedding(current_question)
            
            # Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(query_embedding)
            
            if verbose:
                print(f"\n=== Iteration {iteration}/{max_iterations} ===")
                print(f"Query: {current_question}")
                print("\n=== Relevant Chunks ===")
                for i, chunk in enumerate(chunks):
                    print(f"\n--- Chunk {i+1} (Similarity: {chunk['similarity']:.4f}) ---")
                    print(f"Source: {chunk['file_path']}")
                    print(f"Text: {chunk['chunk_text'][:200]}...")
            
            # Format the context
            context = self.format_context(chunks)
            
            # Generate the answer
            result = self.chain.invoke({"context": context, "question": original_question})
            response = result.get("text", "")  # Extract the answer from the result
            
            logger.info(f"Answer generated with {len(chunks)} relevant chunks")
            
            # Check if the answer contains keywords for a new search
            new_keywords = self._extract_keywords_for_new_search(response)
            
            if not new_keywords:
                # If no new keywords, we have a satisfactory answer or cannot improve
                all_responses.append(f"[Iteration {iteration}] {response}")
                break
            else:
                # Add the current answer to the history
                all_responses.append(f"[Iteration {iteration}] {response}")
                
                # Create a new query with the keywords
                current_question = f"{' '.join(new_keywords)}"
                search_history.append(current_question)
                
                logger.info(f"New keywords identified: {new_keywords}")
        
        # If we reached the maximum number of iterations without a satisfactory answer
        if iteration == max_iterations and new_keywords:
            all_responses.append(
                f"\n\nAfter {max_iterations} iterations, no satisfactory answer could be found. "
                f"Searches were conducted with the following queries: {search_history}"
            )
        
        # Return all responses or only the last one as needed
        if verbose:
            return "\n\n".join(all_responses)
        else:
            return all_responses[-1]
    
    def _extract_keywords_for_new_search(self, response: str) -> List[str]:
        """
        Extract keywords for a new search from the LLM's response.
        
        Args:
            response: LLM's response
            
        Returns:
            List[str]: List of keywords for a new search
        """
        # Look for the keywords section in the response
        if "KEYWORDS_FOR_NEW_SEARCH:" in response:
            try:
                # Extract the part after the indicator
                keywords_part = response.split("KEYWORDS_FOR_NEW_SEARCH:")[1].strip()
                # Clean and split into individual keywords
                keywords = [k.strip() for k in keywords_part.split(",")]
                # Filter out empty keywords
                keywords = [k for k in keywords if k]
                return keywords
            except Exception as e:
                logger.error(f"Error extracting keywords: {e}")
        
        # If no keywords found or error, return an empty list
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="RAG system for answering questions based on indexed documents.")
    parser.add_argument("question", type=str, nargs="?", help="Question to answer (if not provided, interactive mode will be used)")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="Name of the embedding model")
    parser.add_argument("--llm-model", type=str, help="Name of the language model (if not specified, uses OPENROUTER_MODEL)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of search iterations")
    parser.add_argument("--verbose", action="store_true", help="Show process details")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Create RAG system
    rag_system = RAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    print("\n=== RAG System ===")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Language model: {args.llm_model}")
    print(f"Number of documents to retrieve: {args.top_k}")
    print(f"Maximum number of iterations: {args.max_iterations}")    

    # Check if we should run in interactive mode or process a single question
    if args.interactive or (args.question is None):
        print("\nRunning in interactive mode. Type 'exit' to quit.")
        # Interactive loop
        while True:
            print("\n")
            question = input("Question: ")
            
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            if not question.strip():
                continue
            
            try:
                answer = rag_system.answer_question(question, verbose=args.verbose, max_iterations=args.max_iterations)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                print(f"\nError: {e}")
    else:
        # Process the question provided as command line argument
        try:
            answer = rag_system.answer_question(args.question, verbose=args.verbose, max_iterations=args.max_iterations)
            print(answer)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            print(f"Error: {e}")
if __name__ == "__main__":
    main()