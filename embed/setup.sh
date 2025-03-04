#!/bin/bash

# Script d'installation pour le système d'embedding

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Installation du système d'embedding ===${NC}\n"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Erreur: Python 3 n'est pas installé.${NC}"
    echo -e "Veuillez installer Python 3 avant de continuer."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Erreur: pip n'est pas installé.${NC}"
    echo -e "Veuillez installer pip avant de continuer."
    exit 1
fi

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Avertissement: Docker n'est pas installé.${NC}"
    echo -e "Docker est nécessaire pour exécuter PostgreSQL et Ollama."
    echo -e "Veuillez installer Docker avant d'exécuter le système d'embedding."
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Avertissement: Docker Compose n'est pas installé.${NC}"
    echo -e "Docker Compose est nécessaire pour exécuter les services."
    echo -e "Veuillez installer Docker Compose avant d'exécuter le système d'embedding."
fi

echo -e "${GREEN}Création de l'environnement virtuel...${NC}"
python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Installation des dépendances...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Vérification du fichier .env...${NC}"
if [ -f "../.env" ]; then
    echo -e "${GREEN}Le fichier .env existe à la racine du projet.${NC}"
else
    echo -e "${RED}Erreur: Le fichier .env n'existe pas à la racine du projet.${NC}"
    echo -e "${YELLOW}Veuillez créer un fichier .env à la racine du projet avec les paramètres de connexion.${NC}"
    exit 1
fi

echo -e "${GREEN}Rendre les scripts exécutables...${NC}"
chmod +x embed.py
chmod +x test_embed.py
chmod +x rag_example.py

echo -e "\n${GREEN}Installation terminée avec succès!${NC}"
echo -e "\n${BLUE}=== Étapes suivantes ===${NC}"
echo -e "1. Vérifier que le fichier .env à la racine du projet contient les paramètres de connexion corrects"
echo -e "2. Démarrer les services avec: docker-compose up -d"
echo -e "3. Exécuter le script d'embedding avec: ./embed.py"
echo -e "4. Tester le système avec: ./test_embed.py"
echo -e "5. Utiliser le système RAG avec: ./rag_example.py"

echo -e "\n${BLUE}=== Documentation ===${NC}"
echo -e "Pour plus d'informations, consultez le fichier README.md"