#!/bin/bash

# Local RAG System Setup Script
# This script initializes all required services and configurations
# 
# This script is IDEMPOTENT and can be safely run multiple times:
# - Existing services will be updated/restarted if needed
# - Existing models will be skipped
# - Existing collections will be preserved
# - Database initialization only runs on first volume creation

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^\s*$' | xargs)
    echo "✓ Loaded environment variables from .env"
else
    echo "⚠ Warning: .env file not found, using default values"
fi

# Set default values if not provided in .env
: ${N8N_EXTERNAL_PORT:=5678}
: ${OLLAMA_PORT:=11434}
: ${QDRANT_HTTP_PORT:=6333}
: ${POSTGRES_EXTERNAL_PORT:=5436}
: ${OPEN_WEBUI_PORT:=3000}
: ${POSTGRES_USER:=n8n}
: ${POSTGRES_DB:=n8n_rag}
: ${OLLAMA_MODELS:="qllama/multilingual-e5-large-instruct:latest"}
: ${QDRANT_COLLECTION_NAME:=documents}
: ${EMBEDDING_DIMENSION:=1024}

echo "============================================"
echo "  Local RAG System Setup"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}✗ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}✗ Docker daemon is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
}

# Check for existing setup
check_existing_setup() {
    echo -e "\n${BLUE}Checking for existing installation...${NC}"
    
    local existing_services=0
    
    # Check if containers are already running
    if docker ps --format '{{.Names}}' | grep -q "^n8n$"; then
        echo -e "${BLUE}  • n8n is already running${NC}"
        existing_services=$((existing_services + 1))
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
        echo -e "${BLUE}  • Ollama is already running${NC}"
        existing_services=$((existing_services + 1))
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "^qdrant$"; then
        echo -e "${BLUE}  • Qdrant is already running${NC}"
        existing_services=$((existing_services + 1))
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "^postgres$"; then
        echo -e "${BLUE}  • PostgreSQL is already running${NC}"
        existing_services=$((existing_services + 1))
    fi
    
    if [ $existing_services -gt 0 ]; then
        echo -e "${BLUE}  Found $existing_services running service(s). Will update if needed.${NC}"
    else
        echo -e "${BLUE}  No existing services found. Performing fresh installation.${NC}"
    fi
}

# Create documents directory
setup_directories() {
    echo -e "\n${YELLOW}Setting up directories...${NC}"
    
    if [ -d "documents" ]; then
        echo -e "${BLUE}  • documents/ directory already exists${NC}"
    else
        mkdir -p documents
        echo -e "${GREEN}  • Created documents/ directory${NC}"
    fi
    
    # Create .gitkeep to preserve empty directory in git
    touch documents/.gitkeep
    
    echo -e "${GREEN}✓ Directory setup complete${NC}"
}

# Start Docker services
start_services() {
    echo -e "\n${YELLOW}Starting Docker services...${NC}"
    
    if docker compose version &> /dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi
    
    echo -e "${GREEN}✓ Services started${NC}"
}

# Wait for services to be ready
wait_for_services() {
    echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
    
    # Wait for Ollama
    echo "Waiting for Ollama on port ${OLLAMA_PORT}..."
    until curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null 2>&1; do
        sleep 2
    done
    echo -e "${GREEN}✓ Ollama is ready${NC}"
    
    # Wait for Qdrant
    echo "Waiting for Qdrant on port ${QDRANT_HTTP_PORT}..."
    until curl -s http://localhost:${QDRANT_HTTP_PORT}/collections > /dev/null 2>&1; do
        sleep 2
    done
    echo -e "${GREEN}✓ Qdrant is ready${NC}"
    
    # Wait for PostgreSQL
    echo "Waiting for PostgreSQL on port ${POSTGRES_EXTERNAL_PORT}..."
    until docker exec postgres pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} > /dev/null 2>&1; do
        sleep 2
    done
    echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
    
    # Wait for n8n
    echo "Waiting for n8n on port ${N8N_EXTERNAL_PORT}..."
    until curl -s http://localhost:${N8N_EXTERNAL_PORT}/healthz > /dev/null 2>&1; do
        sleep 2
    done
    echo -e "${GREEN}✓ n8n is ready${NC}"
}

# Pull required Ollama models
pull_models() {
    echo -e "\n${YELLOW}Pulling Ollama models (this may take a while on first run)...${NC}"
    echo -e "${BLUE}Models to pull: ${OLLAMA_MODELS}${NC}"
    
    # Function to check if model exists
    check_model() {
        docker exec ollama ollama list | grep -q "$1" 2>/dev/null
    }
    
    # Split OLLAMA_MODELS by comma and process each
    IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS"
    
    for model in "${MODELS[@]}"; do
        # Trim whitespace
        model=$(echo "$model" | xargs)
        
        # Extract model name without tag for checking
        model_name=$(echo "$model" | cut -d':' -f1)
        
        if check_model "$model_name"; then
            echo -e "${BLUE}  • $model already exists, skipping${NC}"
        else
            echo "  Pulling $model (this will take several minutes)..."
            if docker exec ollama ollama pull "$model"; then
                echo -e "${GREEN}  ✓ $model pulled successfully${NC}"
            else
                echo -e "${RED}  ✗ Failed to pull $model${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}✓ Model pulling complete${NC}"
}

# Pre-download HuggingFace tokenizer vocab files for the chunker
# (only ~5 MB SentencePiece vocab — pure CPU, no model weights)
download_tokenizer() {
    local model_id="intfloat/multilingual-e5-large-instruct"
    echo -e "\n${YELLOW}Pre-downloading HF tokenizer for ${model_id}...${NC}"

    # Prefer the project venv; fall back to system python3
    local PYTHON="python3"
    if [ -f ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    fi

    if ! command -v "$PYTHON" &>/dev/null && ! [ -x "$PYTHON" ]; then
        echo -e "${YELLOW}  ⚠ Python not found locally — tokenizer will be downloaded on first backend start${NC}"
        return
    fi

    if "$PYTHON" - <<'EOF'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
print("tokenizer cached")
EOF
    then
        echo -e "${GREEN}  ✓ Tokenizer vocab cached — set HF_HUB_OFFLINE=1 to block outbound Hub calls${NC}"
    else
        echo -e "${YELLOW}  ⚠ Tokenizer pre-download failed — will retry on first backend start${NC}"
    fi
}

# Create Qdrant collection
setup_qdrant() {
    echo -e "\n${YELLOW}Setting up Qdrant collection...${NC}"
    echo -e "${BLUE}Collection: ${QDRANT_COLLECTION_NAME}, Dimension: ${EMBEDDING_DIMENSION}${NC}"
    
    local qdrant_url="http://localhost:${QDRANT_HTTP_PORT}"
    
    # Check if collection exists
    if curl -s "${qdrant_url}/collections/${QDRANT_COLLECTION_NAME}" 2>/dev/null | grep -q '"status":"ok"'; then
        echo -e "${BLUE}  • Collection '${QDRANT_COLLECTION_NAME}' already exists${NC}"
        
        # Get collection info
        local collection_info=$(curl -s "${qdrant_url}/collections/${QDRANT_COLLECTION_NAME}" 2>/dev/null)
        local vectors_count=$(echo "$collection_info" | grep -o '"vectors_count":[0-9]*' | cut -d':' -f2)
        
        if [ -n "$vectors_count" ]; then
            echo -e "${BLUE}  • Current vector count: $vectors_count${NC}"
        fi
    else
        echo "  Creating collection '${QDRANT_COLLECTION_NAME}'..."
        response=$(curl -s -w "\n%{http_code}" -X PUT "${qdrant_url}/collections/${QDRANT_COLLECTION_NAME}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"vectors\": {
                    \"size\": ${EMBEDDING_DIMENSION},
                    \"distance\": \"Cosine\"
                }
            }")
        
        http_code=$(echo "$response" | tail -n1)
        
        if [ "$http_code" = "200" ]; then
            echo -e "${GREEN}  ✓ Qdrant collection '${QDRANT_COLLECTION_NAME}' created${NC}"
        else
            echo -e "${RED}  ✗ Failed to create collection (HTTP $http_code)${NC}"
        fi
    fi
    
    echo -e "${GREEN}✓ Qdrant setup complete${NC}"
}

# Import n8n workflows
import_workflows() {
    echo -e "\n${YELLOW}Importing n8n workflows...${NC}"
    
    echo -e "${YELLOW}Please import the following workflows manually in n8n:${NC}"
    echo "  1. Open http://localhost:${N8N_EXTERNAL_PORT} in your browser"
    echo "  2. Go to Workflows → Import"
    echo "  3. Import local_rag_ingestion.json"
    echo "  4. Import local_rag_retrieval.json"
    echo "  5. Configure credentials for:"
    echo "     - Ollama API (http://ollama:11434)"
    echo "     - Qdrant API (http://qdrant:6333)"
    echo "     - PostgreSQL (host: postgres, db: ${POSTGRES_DB}, user: ${POSTGRES_USER}, password: from .env)"
}

# Print summary
print_summary() {
    echo -e "\n============================================"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo "============================================"
    echo ""
    echo "Service URLs:"
    echo "  • n8n:         http://localhost:${N8N_EXTERNAL_PORT}"
    echo "  • Open WebUI:  http://localhost:${OPEN_WEBUI_PORT}"
    echo "  • Qdrant:      http://localhost:${QDRANT_HTTP_PORT}/dashboard"
    echo "  • Ollama API:  http://localhost:${OLLAMA_PORT}"
    echo ""
    echo "Configuration:"
    echo "  • Qdrant Collection: ${QDRANT_COLLECTION_NAME}"
    echo "  • Embedding Dimension: ${EMBEDDING_DIMENSION}"
    echo "  • PostgreSQL DB: ${POSTGRES_DB}"
    echo ""
    echo "Credentials (from .env):"
    echo "  • n8n: ${N8N_BASIC_AUTH_USER} / [see .env]"
    echo "  • PostgreSQL: ${POSTGRES_USER} / [see .env]"
    echo ""
    echo "Next Steps:"
    echo "  1. Import workflows into n8n"
    echo "  2. Configure n8n credentials"
    echo "  3. Add documents to ${DOCUMENTS_PATH:-./documents} folder"
    echo "  4. Activate the ingestion workflow"
    echo "  5. Test with Open WebUI"
    echo ""
    echo "Documentation: See README.md for detailed instructions"
    echo "To customize settings, edit .env file and re-run ./setup.sh"
}

# Main execution
main() {
    check_prerequisites
    check_existing_setup
    setup_directories
    start_services
    wait_for_services
    pull_models
    download_tokenizer
    setup_qdrant
    import_workflows
    print_summary
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Local RAG System Setup Script (Idempotent)"
        echo ""
        echo "This script can be safely run multiple times:"
        echo "  • Existing services will be updated if configuration changed"
        echo "  • Already downloaded models will be skipped"
        echo "  • Existing Qdrant collections will be preserved"
        echo "  • Database data persists in Docker volumes"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --reset        Remove all volumes and perform fresh installation"
        echo ""
        exit 0
        ;;
    --reset)
        echo -e "${RED}WARNING: This will delete all data (volumes, containers, networks)${NC}"
        read -p "Are you sure? (type 'yes' to confirm): " confirm
        if [ "$confirm" = "yes" ]; then
            echo -e "\n${YELLOW}Stopping and removing containers...${NC}"
            docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null || true
            echo -e "${GREEN}✓ Reset complete. Running fresh installation...${NC}\n"
            main
        else
            echo -e "${BLUE}Reset cancelled.${NC}"
            exit 0
        fi
        ;;
    *)
        main "$@"
        ;;
esac
