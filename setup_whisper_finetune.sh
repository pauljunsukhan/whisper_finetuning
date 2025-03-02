#!/bin/bash
set -e

# Color codes for prettier output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}         Whisper Fine-tuning Environment Setup Script             ${NC}"
echo -e "${BLUE}==================================================================${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    elif command_exists lsb_release; then
        DISTRO=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        DISTRO=$DISTRIB_ID
    elif [ -f /etc/debian_version ]; then
        DISTRO="debian"
    else
        DISTRO="unknown"
    fi
    echo $DISTRO
}

# Check if HF_TOKEN is provided as an argument
if [ "$1" != "" ]; then
    HF_TOKEN="$1"
    echo -e "${GREEN}Hugging Face token provided as argument.${NC}"
else
    # Check if HF_TOKEN environment variable exists
    if [ -z "${HF_TOKEN}" ]; then
        echo -e "${YELLOW}No Hugging Face token found in environment variables.${NC}"
        echo -e "Please enter your Hugging Face token (or press Enter to skip for now):"
        read -r input_token
        
        if [ -n "$input_token" ]; then
            HF_TOKEN="$input_token"
            export HF_TOKEN="$input_token"
            
            # Add to shell configuration for persistence
            echo -e "${YELLOW}Would you like to add the Hugging Face token to your shell configuration? (y/n)${NC}"
            read -r add_to_shell
            
            if [[ "$add_to_shell" =~ ^[Yy]$ ]]; then
                if [ -f "$HOME/.bashrc" ]; then
                    echo "export HF_TOKEN=\"$input_token\"" >> "$HOME/.bashrc"
                    echo -e "${GREEN}Added HF_TOKEN to ~/.bashrc${NC}"
                elif [ -f "$HOME/.zshrc" ]; then
                    echo "export HF_TOKEN=\"$input_token\"" >> "$HOME/.zshrc"
                    echo -e "${GREEN}Added HF_TOKEN to ~/.zshrc${NC}"
                else
                    echo -e "${YELLOW}Could not find .bashrc or .zshrc. Token will only be available for this session.${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}No token provided. You may need to set the HF_TOKEN environment variable later.${NC}"
        fi
    else
        echo -e "${GREEN}Using Hugging Face token from environment variables.${NC}"
    fi
fi

# Install Git LFS
echo -e "\n${BLUE}Checking Git LFS installation...${NC}"
if ! command_exists git-lfs; then
    echo -e "${YELLOW}Git LFS is not installed. Installing...${NC}"
    
    DISTRO=$(detect_distro)
    echo -e "Detected distribution: ${BLUE}$DISTRO${NC}"
    
    case $DISTRO in
        ubuntu|debian|linuxmint|pop)
            sudo apt-get update
            sudo apt-get install -y git-lfs
            ;;
        fedora|centos|rhel)
            sudo dnf install -y git-lfs
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm git-lfs
            ;;
        *)
            echo -e "${YELLOW}Unsupported distribution, installing Git LFS from source...${NC}"
            # Download and install from source for other distros
            TMP_DIR=$(mktemp -d)
            cd "$TMP_DIR" || exit
            curl -sSL https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz -o git-lfs.tar.gz
            tar -xzf git-lfs.tar.gz
            ./install.sh
            cd - || exit
            rm -rf "$TMP_DIR"
            ;;
    esac
    
    echo -e "${GREEN}Git LFS installed successfully.${NC}"
else
    echo -e "${GREEN}Git LFS is already installed.${NC}"
fi

# Initialize Git LFS
echo -e "\n${BLUE}Initializing Git LFS...${NC}"
git lfs install
echo -e "${GREEN}Git LFS initialized.${NC}"

# Check if uv is installed
echo -e "\n${BLUE}Checking uv installation...${NC}"
if ! command_exists uv; then
    echo -e "${YELLOW}uv is not installed. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source shell profiles to make uv available in the current session
    if [ -f ~/.bashrc ]; then
        echo -e "${GREEN}Sourcing ~/.bashrc to update PATH${NC}"
        . ~/.bashrc
    elif [ -f ~/.bash_profile ]; then
        echo -e "${GREEN}Sourcing ~/.bash_profile to update PATH${NC}"
        . ~/.bash_profile
    elif [ -f ~/.zshrc ]; then
        echo -e "${GREEN}Sourcing ~/.zshrc to update PATH${NC}"
        . ~/.zshrc
    fi
    
    # As a fallback, if uv still isn't in PATH, try to add its location directly
    if ! command_exists uv; then
        echo -e "${YELLOW}Adding uv to PATH manually...${NC}"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Verify installation
    if command_exists uv; then
        echo -e "${GREEN}uv installed successfully.${NC}"
    else
        echo -e "${RED}Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}uv is already installed.${NC}"
fi

# Update uv if already installed
if command_exists uv; then
    echo -e "\n${BLUE}Updating uv to the latest version...${NC}"
    uv self update || echo -e "${YELLOW}Could not update uv. Continuing with current version.${NC}"
fi

# Create a virtual environment with the specific Python version
echo -e "\n${BLUE}Creating virtual environment...${NC}"
VENV_NAME="whisper_finetune"

# Try to create the environment with the specific Python version
echo -e "Attempting to create virtual environment with Python 3.9.21..."
if uv venv --python=3.9.21 "$VENV_NAME" 2>/dev/null; then
    echo -e "${GREEN}Virtual environment created with Python 3.9.21.${NC}"
else
    # If that fails, try to find/install Python 3.9
    echo -e "${YELLOW}Python 3.9.21 not found. Attempting to find/install Python 3.9...${NC}"
    
    if command_exists uv; then
        # Try using uv's Python management capabilities
        echo -e "Using uv to install Python 3.9..."
        uv python install 3.9 || echo -e "${YELLOW}Could not install Python 3.9 with uv.${NC}"
        
        # Try creating the environment again with Python 3.9
        if uv venv --python=3.9 "$VENV_NAME" 2>/dev/null; then
            echo -e "${GREEN}Virtual environment created with Python 3.9.${NC}"
        else
            echo -e "${YELLOW}Could not create environment with Python 3.9. Using system Python...${NC}"
            uv venv "$VENV_NAME"
        fi
    else
        echo -e "${YELLOW}Using system Python...${NC}"
        uv venv "$VENV_NAME"
    fi
fi

# Activate the virtual environment
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Display Python version
echo -e "\n${BLUE}Python version in virtual environment:${NC}"
python --version

# Install dependencies using uv
echo -e "\n${BLUE}Installing dependencies using uv...${NC}"
# First check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "Found requirements.txt. Installing dependencies..."
    
    # Use uv's faster sync capabilities instead of simple install if pip-tools compatible
    if grep -q "^#" "requirements.txt" && grep -q "^[a-zA-Z0-9]" "requirements.txt"; then
        echo -e "${YELLOW}Using uv pip sync for better dependency resolution...${NC}"
        uv pip sync requirements.txt
    else
        echo -e "${YELLOW}Using uv pip install for basic installation...${NC}"
        uv pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
else
    echo -e "${RED}requirements.txt not found. Please make sure it exists in the current directory.${NC}"
    exit 1
fi

# Verify installations
echo -e "\n${BLUE}Verifying installations...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
python -c "import evaluate; print(f'Evaluate version: {evaluate.__version__}')"
python -c "import librosa; print(f'Librosa version: {librosa.__version__}')"
python -c "import soundfile; print(f'Soundfile version: {soundfile.__version__}')"

# Validate HuggingFace token if provided
if [ -n "${HF_TOKEN}" ]; then
    echo -e "\n${BLUE}Validating Hugging Face token...${NC}"
    # Install huggingface_hub if not included in requirements
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo -e "${YELLOW}Installing huggingface_hub...${NC}"
        uv pip install huggingface_hub
    fi
    
    # Validate token with a simple Python script
    TOKEN_VALID=$(python -c "
import os
from huggingface_hub import HfApi
try:
    api = HfApi(token='${HF_TOKEN}')
    api.whoami()
    print('valid')
except Exception as e:
    print('invalid')
")
    
    if [ "$TOKEN_VALID" = "valid" ]; then
        echo -e "${GREEN}Hugging Face token is valid.${NC}"
    else
        echo -e "${RED}Hugging Face token validation failed. Please check your token.${NC}"
    fi
fi

echo -e "\n${BLUE}==================================================================${NC}"
echo -e "${GREEN}✅ Setup complete! Virtual environment '${VENV_NAME}' is ready.${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo -e "\n${YELLOW}To activate the environment, run:${NC}"
echo -e "source ${VENV_NAME}/bin/activate"
echo -e "\n${YELLOW}To run the Whisper fine-tuning script:${NC}"
echo -e "python experiments/largev3_004/whisper_experiment_004.py"
echo -e "\n${BLUE}==================================================================${NC}" 