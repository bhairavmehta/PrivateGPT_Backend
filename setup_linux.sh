#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}
██████╗ ██████╗ ██╗██╗   ██╗ █████╗ ████████╗███████╗     ██████╗ ██████╗ ████████╗
██╔══██╗██╔══██╗██║██║   ██║██╔══██╗╚══██╔══╝██╔════╝    ██╔════╝ ██╔══██╗╚══██╔══╝
██████╔╝██████╔╝██║██║   ██║███████║   ██║   █████╗      ██║  ███╗██████╔╝   ██║   
██╔═══╝ ██╔══██╗██║╚██╗ ██╔╝██╔══██║   ██║   ██╔══╝      ██║   ██║██╔═══╝    ██║   
██║     ██║  ██║██║ ╚████╔╝ ██║  ██║   ██║   ███████╗    ╚██████╔╝██║        ██║   
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚══════╝     ╚═════╝ ╚═╝        ╚═╝   
                                                                                     
██╗     ██╗███╗   ██╗██╗   ██╗██╗  ██╗    ███████╗███████╗████████╗██╗   ██╗██████╗ 
██║     ██║████╗  ██║██║   ██║╚██╗██╔╝    ██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗
██║     ██║██╔██╗ ██║██║   ██║ ╚███╔╝     ███████╗█████╗     ██║   ██║   ██║██████╔╝
██║     ██║██║╚██╗██║██║   ██║ ██╔██╗     ╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝ 
███████╗██║██║ ╚████║╚██████╔╝██╔╝ ██╗    ███████║███████╗   ██║   ╚██████╔╝██║     
╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝    ╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     
${NC}"
    echo -e "${CYAN}🚀 Comprehensive Linux Setup for Private GPT Backend${NC}"
    echo -e "${CYAN}🤖 Installing system dependencies, Python packages, and downloading models${NC}"
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect Linux distribution"
        exit 1
    fi
    print_status "Detected Linux distribution: $DISTRO $VERSION"
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Update package lists
    print_status "Updating package lists..."
    
    if command_exists apt-get; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            pkg-config \
            libopenblas-dev \
            liblapack-dev \
            libeigen3-dev \
            curl \
            wget \
            git \
            python3 \
            python3-pip \
            python3-dev \
            python3-venv \
            python3-wheel \
            python3-setuptools \
            ffmpeg \
            libsm6 \
            libxext6 \
            libfontconfig1 \
            libxrender1 \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libgtk-3-0 \
            libqt5gui5 \
            libnss3-dev \
            libxss1 \
            libxcomposite1 \
            libxcursor1 \
            libxdamage1 \
            libxfixes3 \
            libxi6 \
            libxrandr2 \
            libasound2 \
            libpangocairo-1.0-0 \
            libatk1.0-0 \
            libcairo-gobject2 \
            libgtk-3-0 \
            libgdk-pixbuf2.0-0 \
            unzip \
            software-properties-common \
            apt-transport-https \
            ca-certificates \
            gnupg \
            lsb-release
            
        # Install Google Chrome for Selenium web scraping
        print_status "Installing Google Chrome for web scraping..."
        if ! command_exists google-chrome; then
            # Add Google Chrome repository
            wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
            echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
            sudo apt-get update
            sudo apt-get install -y google-chrome-stable
            print_success "Google Chrome installed successfully!"
        else
            print_success "Google Chrome already installed!"
        fi
            
    elif command_exists yum; then
        # CentOS/RHEL/Fedora
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake \
            pkgconfig \
            openblas-devel \
            lapack-devel \
            eigen3-devel \
            curl \
            wget \
            git \
            python3 \
            python3-pip \
            python3-devel \
            ffmpeg \
            libSM \
            libXext \
            fontconfig \
            libXrender \
            mesa-libGL \
            glib2 \
            gtk3 \
            qt5-qtbase-gui \
            nss-devel \
            libXScrnSaver \
            libXcomposite \
            libXcursor \
            libXdamage \
            libXfixes \
            libXi \
            libXrandr \
            alsa-lib \
            pango \
            atk \
            cairo-gobject \
            gtk3 \
            gdk-pixbuf2 \
            unzip
            
        # Install Google Chrome for Selenium web scraping
        print_status "Installing Google Chrome for web scraping..."
        if ! command_exists google-chrome; then
            # Add Google Chrome repository for RHEL/CentOS/Fedora
            cat << 'EOF' | sudo tee /etc/yum.repos.d/google-chrome.repo
[google-chrome]
name=google-chrome
baseurl=http://dl.google.com/linux/chrome/rpm/stable/x86_64
enabled=1
gpgcheck=1
gpgkey=https://dl.google.com/linux/linux_signing_key.pub
EOF
            sudo yum install -y google-chrome-stable
            print_success "Google Chrome installed successfully!"
        else
            print_success "Google Chrome already installed!"
        fi
            
    elif command_exists pacman; then
        # Arch Linux
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm \
            base-devel \
            cmake \
            pkgconf \
            openblas \
            lapack \
            eigen \
            curl \
            wget \
            git \
            python \
            python-pip \
            ffmpeg \
            libsm \
            libxext \
            fontconfig \
            libxrender \
            mesa \
            glib2 \
            gtk3 \
            qt5-base \
            nss \
            libxss \
            libxcomposite \
            libxcursor \
            libxdamage \
            libxfixes \
            libxi \
            libxrandr \
            alsa-lib \
            pango \
            atk \
            cairo \
            gtk3 \
            gdk-pixbuf2 \
            unzip
            
        # Install Google Chrome for Selenium web scraping
        print_status "Installing Google Chrome for web scraping..."
        if ! command_exists google-chrome; then
            # For Arch Linux, install from AUR (requires yay or manual installation)
            if command_exists yay; then
                yay -S --noconfirm google-chrome
                print_success "Google Chrome installed successfully!"
            else
                print_warning "AUR helper 'yay' not found. Installing google-chrome manually..."
                # Install google-chrome from AUR manually
                cd /tmp
                git clone https://aur.archlinux.org/google-chrome.git
                cd google-chrome
                makepkg -si --noconfirm
                cd ..
                rm -rf google-chrome
                print_success "Google Chrome installed successfully!"
            fi
        else
            print_success "Google Chrome already installed!"
        fi
    else
        print_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    print_success "System dependencies installed successfully!"
}

# Function to install Node.js and npm
install_nodejs() {
    print_status "Installing Node.js and npm..."
    
    if command_exists node && command_exists npm; then
        NODE_VERSION=$(node --version)
        print_success "Node.js already installed: $NODE_VERSION"
    else
        # Install Node.js using NodeSource repository for latest LTS
        print_status "Installing Node.js LTS..."
        
        if command_exists apt-get; then
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command_exists yum; then
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
            sudo yum install -y nodejs npm
        elif command_exists pacman; then
            sudo pacman -S --noconfirm nodejs npm
        fi
        
        print_success "Node.js and npm installed successfully!"
    fi
    
    # Install Bun (optional, as the project has bun.lockb)
    if ! command_exists bun; then
        print_status "Installing Bun (JavaScript runtime)..."
        curl -fsSL https://bun.sh/install | bash
        export PATH="$HOME/.bun/bin:$PATH"
        echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.bashrc
        print_success "Bun installed successfully!"
    else
        print_success "Bun already installed"
    fi
}

# Function to create Python requirements.txt
create_requirements_txt() {
    print_status "Creating Python requirements.txt file..."
    
    cat > requirements.txt << 'EOF'
# Core FastAPI and web framework
fastapi==0.115.4
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic-settings==2.6.0

# GGUF model loading and AI
llama-cpp-python==0.3.1
torch>=2.0.0
transformers>=4.45.0
qwen-vl-utils[decord]==0.0.8

# Document processing
PyPDF2==3.0.1
python-docx==1.1.2
python-magic==0.4.27
fitz==0.0.1.dev2
PyMuPDF==1.24.0
docx2txt==0.8
mammoth==1.8.0
reportlab==4.2.5
Pillow==10.4.0
markdown==3.7

# Web scraping and search
selenium==4.26.1
beautifulsoup4==4.12.3
webdriver-manager==4.0.2
undetected-chromedriver==3.5.4
httpx==0.27.2
duckduckgo-search==6.3.0
requests==2.32.3

# Data processing and visualization
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1
wordcloud==1.9.3
networkx==3.4

# Computer vision and image processing
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84

# Template processing
jinja2==3.1.4

# Utility libraries
psutil==6.1.0
asyncio-throttle==1.0.2
aiofiles==24.1.0
python-dateutil==2.9.0.post0
pathlib-mate==1.3.2
typing-extensions==4.12.2

# Database (SQLite for embeddings)
sqlite3

# Development and testing
pytest==8.3.3
black==24.10.0
flake8==7.1.1
isort==5.13.2

# Additional scientific computing
scipy==1.14.1
scikit-learn==1.5.2

# Audio processing (if needed)
soundfile==0.12.1
librosa==0.10.2

# Environment and configuration
python-dotenv==1.0.1

# Concurrent processing
concurrent-futures==3.1.1

# Additional utilities for file handling
chardet==5.2.0
mimetypes-plus==1.0.0

# For advanced chart generation
bokeh==3.6.0
altair==5.4.1

# For enhanced document processing
openpyxl==3.1.5
xlsxwriter==3.2.0
EOF

    print_success "requirements.txt created successfully!"
}

# Function to setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created!"
    else
        print_success "Virtual environment already exists!"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        create_requirements_txt
    fi
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    print_warning "This may take several minutes, especially for torch and llama-cpp-python..."
    
    # Install llama-cpp-python with CUDA support if CUDA is available
    if command_exists nvcc; then
        print_status "CUDA detected, installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    else
        print_status "CUDA not detected, installing CPU-only llama-cpp-python..."
        pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    fi
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed successfully!"
}

# Function to setup frontend dependencies
setup_frontend() {
    print_status "Setting up frontend dependencies..."
    
    cd frontend
    
    # Install npm dependencies
    if [ -f "package-lock.json" ]; then
        print_status "Installing npm dependencies..."
        npm ci
    elif [ -f "bun.lockb" ] && command_exists bun; then
        print_status "Installing bun dependencies..."
        bun install
    else
        print_status "Installing npm dependencies (first time)..."
        npm install
    fi
    
    cd ..
    print_success "Frontend dependencies installed successfully!"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p models/report_generation
    mkdir -p models/coding
    mkdir -p models/vision
    mkdir -p models/embedding
    mkdir -p templates
    mkdir -p temp_files
    mkdir -p temp_files/charts
    mkdir -p logs
    mkdir -p scripts
    
    print_success "Directories created successfully!"
}

# Function to download models
download_models() {
    print_status "Downloading AI models..."
    print_warning "This will download large files (several GBs). Ensure you have good internet connection!"
    
    # Report Generation Model
    if [ ! -f "models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf" ]; then
        print_status "Downloading Report Generation Model (Qwen2.5-7B-Instruct-Q6_K.gguf)..."
        print_status "File size: ~5.5GB - This may take 10-30 minutes depending on your connection..."
        wget -c "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q6_K.gguf" \
             -O "models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf"
        print_success "Report Generation model downloaded successfully!"
    else
        print_success "Report Generation model already exists!"
    fi
    
    # Coding Model
    if [ ! -f "models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf" ]; then
        print_status "Downloading Coding Model (qwen2.5-coder-7b-instruct-q6_k.gguf)..."
        print_status "File size: ~5.5GB - This may take 10-30 minutes depending on your connection..."
        wget -c "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q6_k.gguf" \
             -O "models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf"
        print_success "Coding model downloaded successfully!"
    else
        print_success "Coding model already exists!"
    fi
    
    # Optional: Vision Model (commented out as it's large and not in the required list)
    # if [ ! -f "models/vision/qwen2.5-vl-7b-instruct-q4_k_m.gguf" ]; then
    #     print_status "Downloading Vision Model (optional)..."
    #     wget -c "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/qwen2.5-vl-7b-instruct-q4_k_m.gguf" \
    #          -O "models/vision/qwen2.5-vl-7b-instruct-q4_k_m.gguf"
    #     print_success "Vision model downloaded successfully!"
    # fi
    
    print_success "All models downloaded successfully!"
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# API Configuration
DEBUG=false
API_TITLE="Private GPT Backend"
API_VERSION="1.0.0"

# Model Paths
MODELS_BASE_PATH="./models/"
TEMP_FILES_PATH="./temp_files/"
TEMPLATES_PATH="./templates/"

# Web Search (optional - add your own keys)
BRAVE_SEARCH_API_KEY=""
OPENAI_API_KEY=""

# Security
ALLOWED_HOSTS=["localhost", "127.0.0.1"]
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Performance
MAX_FILE_SIZE=104857600  # 100MB
MAX_WORKERS=4
MODEL_CACHE_SIZE=2
EOF
        print_success "Environment file created! Please edit .env if needed."
    else
        print_success "Environment file already exists!"
    fi
}

# Function to set permissions
set_permissions() {
    print_status "Setting appropriate permissions..."
    
    # Make scripts executable
    chmod +x setup_linux.sh
    chmod +x scripts/*.py 2>/dev/null || true
    
    # Set directory permissions
    chmod -R 755 models/ templates/ temp_files/ logs/ 2>/dev/null || true
    
    print_success "Permissions set successfully!"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    source venv/bin/activate
    
    print_status "Testing critical Python imports..."
    python3 -c "
import fastapi
import llama_cpp
import torch
import pandas
import matplotlib
import PIL
print('✅ All critical Python packages imported successfully!')
"
    
    # Test Node.js
    if command_exists node && command_exists npm; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_success "Node.js $NODE_VERSION and npm $NPM_VERSION are working!"
    fi
    
    # Test Chrome installation for Selenium
    if command_exists google-chrome; then
        CHROME_VERSION=$(google-chrome --version 2>/dev/null || echo "Google Chrome (version detection failed)")
        print_success "Google Chrome installed: $CHROME_VERSION"
    else
        print_warning "Google Chrome not found! Web scraping may not work properly."
    fi
    
    # Check model files
    if [ -f "models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf" ]; then
        print_success "Report generation model file exists!"
    else
        print_warning "Report generation model file not found!"
    fi
    
    if [ -f "models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf" ]; then
        print_success "Coding model file exists!"
    else
        print_warning "Coding model file not found!"
    fi
    
    print_success "Installation test completed!"
}

# Function to display final instructions
show_final_instructions() {
    print_header
    echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}📋 Quick Start Instructions:${NC}"
    echo ""
    echo -e "${YELLOW}1. Backend (API Server):${NC}"
    echo "   cd /path/to/private_gpt_backend"
    echo "   source venv/bin/activate"
    echo "   uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    echo -e "${YELLOW}2. Frontend (React App):${NC}"
    echo "   cd frontend"
    echo "   npm run dev"
    echo "   # OR if using Bun:"
    echo "   # bun run dev"
    echo ""
    echo -e "${YELLOW}3. Access your application:${NC}"
    echo "   🌐 Frontend: http://localhost:3000"
    echo "   🔌 Backend API: http://localhost:8000"
    echo "   📖 API Docs: http://localhost:8000/docs"
    echo ""
    echo -e "${YELLOW}4. Using Docker (alternative):${NC}"
    echo "   docker-compose up -d"
    echo ""
    echo -e "${CYAN}📁 Downloaded Models:${NC}"
    echo "   📊 Report Generation: models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf"
    echo "   💻 Coding: models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf"
    echo ""
    echo -e "${CYAN}🔧 Configuration:${NC}"
    echo "   Edit .env file to customize settings"
    echo "   Add API keys for web search if needed"
    echo ""
    echo -e "${CYAN}🌐 Web Scraping:${NC}"
    echo "   Google Chrome installed for Selenium web scraping"
    echo "   Supports headless browsing and content extraction"
    echo "   WebDriver automatically managed by undetected-chromedriver"
    echo ""
    echo -e "${GREEN}🚀 Your Private GPT Backend is ready to use!${NC}"
    echo ""
}

# Main installation function
main() {
    print_header
    
    # Check if running as root (not recommended for development)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. Some operations will be adapted for root user."
    fi
    
    # Detect Linux distribution
    detect_distro
    
    # Check for minimum requirements
    print_status "Checking minimum requirements..."
    
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Start installation process
    echo ""
    print_status "Starting installation process..."
    echo ""
    
    # Step 1: Install system dependencies
    install_system_dependencies
    
    # Step 2: Install Node.js
    install_nodejs
    
    # Step 3: Create directories
    create_directories
    
    # Step 4: Setup Python environment
    setup_python_environment
    
    # Step 5: Setup frontend
    setup_frontend
    
    # Step 6: Download models
    download_models
    
    # Step 7: Create environment file
    create_env_file
    
    # Step 8: Set permissions
    set_permissions
    
    # Step 9: Test installation
    test_installation
    
    # Step 10: Show final instructions
    show_final_instructions
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Private GPT Backend Linux Setup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --models-only       Download models only"
        echo "  --no-models         Skip model download"
        echo "  --test              Test current installation"
        echo ""
        echo "This script will:"
        echo "  1. Install system dependencies"
        echo "  2. Setup Python virtual environment"
        echo "  3. Install Python packages"
        echo "  4. Setup Node.js frontend"
        echo "  5. Download AI models (~11GB total)"
        echo "  6. Configure the application"
        echo ""
        exit 0
        ;;
    --models-only)
        print_header
        create_directories
        download_models
        exit 0
        ;;
    --no-models)
        print_header
        detect_distro
        install_system_dependencies
        install_nodejs
        create_directories
        setup_python_environment
        setup_frontend
        create_env_file
        set_permissions
        test_installation
        print_success "Installation completed without downloading models!"
        exit 0
        ;;
    --test)
        print_header
        test_installation
        exit 0
        ;;
    *)
        main
        ;;
esac 