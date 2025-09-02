# setup_windows.ps1
# Comprehensive Windows Setup Script for Private GPT Backend
# Updated with Qwen2.5-14B models
# Requires PowerShell 5.1 or higher and Administrator privileges for some operations

param(
    [switch]$ModelsOnly,
    [switch]$NoModels,
    [switch]$Test,
    [switch]$Help
)

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Magenta = "Magenta"
    Cyan = "Cyan"
}

function Write-ColorOutput {
    param($Message, $Color = "White", $Prefix = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
    Write-Host "[$Prefix] " -NoNewline -ForegroundColor $Color
    Write-Host $Message
}

function Write-Success { param($Message) Write-ColorOutput $Message $Colors.Green "SUCCESS" }
function Write-Warning { param($Message) Write-ColorOutput $Message $Colors.Yellow "WARNING" }
function Write-Error { param($Message) Write-ColorOutput $Message $Colors.Red "ERROR" }
function Write-Info { param($Message) Write-ColorOutput $Message $Colors.Blue "INFO" }

function Write-Header {
    Write-Host @"

██████╗ ██████╗ ██╗██╗   ██╗ █████╗ ████████╗███████╗     ██████╗ ██████╗ ████████╗
██╔══██╗██╔══██╗██║██║   ██║██╔══██╗╚══██╔══╝██╔════╝    ██╔════╝ ██╔══██╗╚══██╔══╝
██████╔╝██████╔╝██║██║   ██║███████║   ██║   █████╗      ██║  ███╗██████╔╝   ██║   
██╔═══╝ ██╔══██╗██║╚██╗ ██╔╝██╔══██║   ██║   ██╔══╝      ██║   ██║██╔═══╝    ██║   
██║     ██║  ██║██║ ╚████╔╝ ██║  ██║   ██║   ███████╗    ╚██████╔╝██║        ██║   
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚══════╝     ╚═════╝ ╚═╝        ╚═╝   

██╗    ██╗██╗███╗   ██╗██████╗  ██████╗ ██╗    ██╗███████╗    ███████╗███████╗████████╗██╗   ██╗██████╗ 
██║    ██║██║████╗  ██║██╔══██╗██╔═══██╗██║    ██║██╔════╝    ██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗
██║ █╗ ██║██║██╔██╗ ██║██║  ██║██║   ██║██║ █╗ ██║███████╗    ███████╗█████╗     ██║   ██║   ██║██████╔╝
██║███╗██║██║██║╚██╗██║██║  ██║██║   ██║██║███╗██║╚════██║    ╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝ 
╚███╔███╔╝██║██║ ╚████║██████╔╝╚██████╔╝╚███╔███╔╝███████║    ███████║███████╗   ██║   ╚██████╔╝██║     
 ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝  ╚══╝╚══╝ ╚══════╝    ╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     

"@ -ForegroundColor Magenta
    
    Write-Host "🚀 Comprehensive Windows Setup for Private GPT Backend" -ForegroundColor Cyan
    Write-Host "🤖 Installing system dependencies, Python packages, and downloading 14B models" -ForegroundColor Cyan
    Write-Host ""
}

function Test-IsAdmin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-CommandExists {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Install-Chocolatey {
    Write-Info "Installing Chocolatey package manager..."
    
    if (Test-CommandExists "choco") {
        Write-Success "Chocolatey already installed!"
        return
    }
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed successfully!"
    }
    catch {
        Write-Error "Failed to install Chocolatey: $($_.Exception.Message)"
        Write-Info "Please install Chocolatey manually from https://chocolatey.org/install"
        exit 1
    }
}

function Install-SystemDependencies {
    Write-Info "Installing system dependencies..."
    
    # Check if running as administrator for Chocolatey installs
    if (-not (Test-IsAdmin)) {
        Write-Warning "Not running as Administrator. Some installations may require elevation."
        Write-Info "You may need to restart PowerShell as Administrator for package installations."
    }
    
    # Install Chocolatey if not present
    Install-Chocolatey
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Essential packages
    $packages = @(
        "python",           # Python 3.x
        "nodejs",           # Node.js and npm
        "git",              # Git version control
        "googlechrome",     # Chrome for web scraping
        "7zip",             # File compression
        "curl",             # Download utility
        "wget"              # Alternative download utility
    )
    
    # Install packages via Chocolatey
    foreach ($package in $packages) {
        Write-Info "Installing $package..."
        try {
            if (Test-IsAdmin) {
                choco install $package -y --no-progress
            } else {
                Write-Warning "Skipping $package installation - requires Administrator privileges"
                Write-Info "Please run: choco install $package -y"
            }
        }
        catch {
            Write-Warning "Failed to install $package via Chocolatey: $($_.Exception.Message)"
        }
    }
    
    # Install Visual Studio Build Tools for Python compilation
    Write-Info "Installing Visual Studio Build Tools for Python packages..."
    try {
        if (Test-IsAdmin) {
            choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools" -y
        } else {
            Write-Warning "Skipping Visual Studio Build Tools - requires Administrator privileges"
            Write-Info "For Python package compilation, install Visual Studio Build Tools manually"
        }
    }
    catch {
        Write-Warning "Failed to install Visual Studio Build Tools"
    }
    
    # Check if Python is available
    if (Test-CommandExists "python") {
        $pythonVersion = python --version 2>&1
        Write-Success "Python installed: $pythonVersion"
    } else {
        Write-Warning "Python not found in PATH. Please ensure Python is installed and added to PATH."
    }
    
    # Check if Node.js is available
    if (Test-CommandExists "node") {
        $nodeVersion = node --version
        $npmVersion = npm --version
        Write-Success "Node.js installed: $nodeVersion, npm: $npmVersion"
    } else {
        Write-Warning "Node.js not found in PATH. Please ensure Node.js is installed and added to PATH."
    }
    
    Write-Success "System dependencies installation completed!"
}

function Install-Python {
    Write-Info "Setting up Python environment..."
    
    # Check Python installation
    if (-not (Test-CommandExists "python")) {
        Write-Error "Python not found. Please install Python 3.8+ from python.org or use 'choco install python'"
        return $false
    }
    
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    return $true
}

function Create-PythonRequirements {
    Write-Info "Creating Python requirements.txt file..."
    
    $requirements = @"
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
python-magic-bin==0.4.14
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

# Additional utilities for file handling
chardet==5.2.0

# For advanced chart generation
bokeh==3.6.0
altair==5.4.1

# For enhanced document processing
openpyxl==3.1.5
xlsxwriter==3.2.0
"@

    $requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8
    Write-Success "requirements.txt created successfully!"
}

function Setup-PythonEnvironment {
    Write-Info "Setting up Python virtual environment..."
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Info "Creating Python virtual environment..."
        python -m venv venv
        Write-Success "Virtual environment created!"
    } else {
        Write-Success "Virtual environment already exists!"
    }
    
    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip in virtual environment
    Write-Info "Upgrading pip in virtual environment..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Create requirements.txt if it doesn't exist
    if (-not (Test-Path "requirements.txt")) {
        Create-PythonRequirements
    }
    
    # Install Python dependencies
    Write-Info "Installing Python dependencies..."
    Write-Warning "This may take several minutes, especially for torch and llama-cpp-python..."
    
    try {
        # Install llama-cpp-python first (often the most problematic)
        Write-Info "Installing llama-cpp-python..."
        python -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
        
        # Install other requirements
        Write-Info "Installing remaining Python packages..."
        python -m pip install -r requirements.txt
        
        Write-Success "Python dependencies installed successfully!"
    }
    catch {
        Write-Error "Failed to install Python dependencies: $($_.Exception.Message)"
        Write-Info "You may need to install Visual Studio Build Tools for C++ compilation"
        return $false
    }
    
    return $true
}

function Setup-Frontend {
    Write-Info "Setting up frontend dependencies..."
    
    if (-not (Test-Path "frontend")) {
        Write-Warning "Frontend directory not found. Creating basic structure..."
        New-Item -ItemType Directory -Name "frontend" -Force
        return
    }
    
    Push-Location "frontend"
    
    try {
        # Install npm dependencies
        if (Test-Path "package-lock.json") {
            Write-Info "Installing npm dependencies from package-lock.json..."
            npm ci
        } elseif (Test-Path "package.json") {
            Write-Info "Installing npm dependencies..."
            npm install
        } else {
            Write-Warning "No package.json found in frontend directory"
        }
        
        Write-Success "Frontend dependencies installed successfully!"
    }
    catch {
        Write-Error "Failed to install frontend dependencies: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
}

function Create-Directories {
    Write-Info "Creating necessary directories..."
    
    $directories = @(
        "models\report_generation",
        "models\coding", 
        "models\vision",
        "models\embedding",
        "templates",
        "temp_files",
        "temp_files\charts",
        "logs",
        "scripts"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
    
    Write-Success "Directories created successfully!"
}

function Download-ModelFile {
    param($Url, $OutputPath, $Description, $ExpectedSizeGB)
    
    Write-Info "Downloading $Description..."
    Write-Info "URL: $Url"
    Write-Info "Destination: $OutputPath"
    Write-Warning "File size: ~$ExpectedSizeGB GB - This may take 10-60 minutes depending on your connection..."
    
    try {
        # Create directory if it doesn't exist
        $outputDir = Split-Path $OutputPath -Parent
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        # Use curl for download with progress
        if (Test-CommandExists "curl") {
            curl -L --progress-bar -o "$OutputPath" "$Url"
        } else {
            # Fallback to Invoke-WebRequest
            $ProgressPreference = 'Continue'
            Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        }
        
        # Verify file size
        if (Test-Path $OutputPath) {
            $fileSizeGB = (Get-Item $OutputPath).Length / 1GB
            Write-Success "Download completed! File size: $([math]::Round($fileSizeGB, 2)) GB"
            
            # Check if size is reasonable (within 20% of expected)
            $sizeDiff = [math]::Abs($fileSizeGB - $ExpectedSizeGB) / $ExpectedSizeGB
            if ($sizeDiff -gt 0.2) {
                Write-Warning "File size differs significantly from expected. Expected: $ExpectedSizeGB GB, Got: $([math]::Round($fileSizeGB, 2)) GB"
            }
            
            return $true
        } else {
            Write-Error "Download failed - file not found"
            return $false
        }
    }
    catch {
        Write-Error "Download failed: $($_.Exception.Message)"
        return $false
    }
}

function Download-Models {
    Write-Info "Downloading AI models (14B parameters)..."
    Write-Warning "This will download large files (several GBs). Ensure you have good internet connection and sufficient disk space!"
    
    # Check available disk space
    $drive = Get-PSDrive -Name C
    $freeSpaceGB = $drive.Free / 1GB
    $requiredSpaceGB = 18  # Approximate total space needed
    
    Write-Info "Available disk space: $([math]::Round($freeSpaceGB, 1)) GB"
    Write-Info "Required space: ~$requiredSpaceGB GB"
    
    if ($freeSpaceGB -lt $requiredSpaceGB) {
        Write-Error "Insufficient disk space! Need at least $requiredSpaceGB GB free."
        return $false
    }
    
    # Model configurations - Updated to 14B models
    $models = @{
        "Report Generation" = @{
            Path = "models\report_generation\Qwen2.5-14B-Instruct-Q6_K.gguf"
            Url = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q6_K.gguf"
            SizeGB = 8.5
        }
        "Coding" = @{
            Path = "models\coding\Qwen2.5-Coder-14B-Instruct-Q6_K.gguf"
            Url = "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf"
            SizeGB = 8.5
        }
        "Embedding" = @{
            Path = "models\embedding\Qwen3-Embedding-0.6B-Q8_0.gguf"
            Url = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
            SizeGB = 0.6
        }
    }
    
    $successCount = 0
    foreach ($modelName in $models.Keys) {
        $model = $models[$modelName]
        
        if (Test-Path $model.Path) {
            Write-Success "$modelName model already exists!"
            $successCount++
            continue
        }
        
        Write-Info "Downloading $modelName model..."
        if (Download-ModelFile -Url $model.Url -OutputPath $model.Path -Description "$modelName Model (14B)" -ExpectedSizeGB $model.SizeGB) {
            Write-Success "$modelName model downloaded successfully!"
            $successCount++
        } else {
            Write-Error "$modelName model download failed!"
        }
        
        Write-Host ""
    }
    
    Write-Info "Model download summary: $successCount/$($models.Count) models downloaded successfully"
    return $successCount -eq $models.Count
}

function Create-EnvironmentFile {
    Write-Info "Creating environment configuration..."
    
    if (Test-Path ".env") {
        Write-Success "Environment file already exists!"
        return
    }
    
    $envContent = @"
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

# Windows specific paths
TEMP_FILES_PATH="./temp_files/"
LOGS_PATH="./logs/"
"@

    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Success "Environment file created! Please edit .env if needed."
}

function Test-Installation {
    Write-Info "Testing installation..."
    
    # Test Python imports
    Write-Info "Testing critical Python imports..."
    
    $pythonTest = @"
try:
    import fastapi
    import llama_cpp
    import torch
    import pandas
    import matplotlib
    import PIL
    print('✅ All critical Python packages imported successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"@
    
    try {
        $pythonTest | python
        Write-Success "Python packages test passed!"
    }
    catch {
        Write-Error "Python packages test failed!"
        return $false
    }
    
    # Test Node.js
    if (Test-CommandExists "node") {
        $nodeVersion = node --version
        $npmVersion = npm --version
        Write-Success "Node.js $nodeVersion and npm $npmVersion are working!"
    } else {
        Write-Warning "Node.js not found!"
    }
    
    # Test Chrome for Selenium
    if (Test-CommandExists "chrome") {
        Write-Success "Google Chrome found for web scraping!"
    } else {
        Write-Warning "Google Chrome not found! Web scraping may not work properly."
    }
    
    # Check model files
    $modelChecks = @(
        "models\report_generation\Qwen2.5-14B-Instruct-Q6_K.gguf",
        "models\coding\Qwen2.5-Coder-14B-Instruct-Q6_K.gguf",
        "models\embedding\Qwen3-Embedding-0.6B-Q8_0.gguf"
    )
    
    foreach ($modelPath in $modelChecks) {
        if (Test-Path $modelPath) {
            $modelName = Split-Path $modelPath -Leaf
            Write-Success "Model file exists: $modelName"
        } else {
            $modelName = Split-Path $modelPath -Leaf
            Write-Warning "Model file missing: $modelName"
        }
    }
    
    Write-Success "Installation test completed!"
    return $true
}

function Show-FinalInstructions {
    Write-Header
    Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Quick Start Instructions:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Backend (API Server):" -ForegroundColor Yellow
    Write-Host "   cd $(Get-Location)"
    Write-Host "   .\venv\Scripts\Activate.ps1"
    Write-Host "   uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
    Write-Host ""
    Write-Host "2. Frontend (React App):" -ForegroundColor Yellow
    Write-Host "   cd frontend"
    Write-Host "   npm run dev"
    Write-Host ""
    Write-Host "3. Access your application:" -ForegroundColor Yellow
    Write-Host "   🌐 Frontend: http://localhost:3000"
    Write-Host "   🔌 Backend API: http://localhost:8000"
    Write-Host "   📖 API Docs: http://localhost:8000/docs"
    Write-Host ""
    Write-Host "4. Using Docker (alternative):" -ForegroundColor Yellow
    Write-Host "   docker-compose up -d"
    Write-Host ""
    Write-Host "📁 Downloaded Models:" -ForegroundColor Cyan
    Write-Host "   📊 Report Generation: models\report_generation\Qwen2.5-14B-Instruct-Q6_K.gguf"
    Write-Host "   💻 Coding: models\coding\Qwen2.5-Coder-14B-Instruct-Q6_K.gguf"
    Write-Host "   🔍 Embedding: models\embedding\Qwen3-Embedding-0.6B-Q8_0.gguf"
    Write-Host ""
    Write-Host "🔧 Configuration:" -ForegroundColor Cyan
    Write-Host "   Edit .env file to customize settings"
    Write-Host "   Add API keys for web search if needed"
    Write-Host ""
    Write-Host "🌐 Web Scraping:" -ForegroundColor Cyan
    Write-Host "   Google Chrome installed for Selenium web scraping"
    Write-Host "   Supports headless browsing and content extraction"
    Write-Host "   WebDriver automatically managed by undetected-chromedriver"
    Write-Host ""
    Write-Host "🚀 Your Private GPT Backend is ready to use!" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Troubleshooting Tips:" -ForegroundColor Yellow
    Write-Host "   - If Python packages fail to install, ensure Visual Studio Build Tools are installed"
    Write-Host "   - For CUDA support, install CUDA toolkit and reinstall llama-cpp-python"
    Write-Host "   - Run PowerShell as Administrator for system-level installations"
    Write-Host "   - Check Windows Defender/Antivirus if downloads are blocked"
    Write-Host "   - 14B models require 16-32GB RAM for optimal performance"
    Write-Host ""
    Write-Host "⚡ Performance Notes:" -ForegroundColor Magenta
    Write-Host "   - 14B models provide significantly better quality than 7B versions"
    Write-Host "   - Recommended: 32GB RAM, RTX 4080+ GPU for best performance"
    Write-Host "   - CPU inference possible but will be slower"
    Write-Host ""
}

function Show-Help {
    Write-Host "Private GPT Backend Windows Setup Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\setup_windows.ps1 [OPTIONS]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -Help               Show this help message"
    Write-Host "  -ModelsOnly         Download models only"
    Write-Host "  -NoModels           Skip model download"
    Write-Host "  -Test               Test current installation"
    Write-Host ""
    Write-Host "This script will:" -ForegroundColor Cyan
    Write-Host "  1. Install system dependencies via Chocolatey"
    Write-Host "  2. Setup Python virtual environment"
    Write-Host "  3. Install Python packages"
    Write-Host "  4. Setup Node.js frontend"
    Write-Host "  5. Download AI models (~18GB total - 14B parameter models)"
    Write-Host "  6. Configure the application"
    Write-Host ""
    Write-Host "Requirements:" -ForegroundColor Yellow
    Write-Host "  - Windows 10/11"
    Write-Host "  - PowerShell 5.1+"
    Write-Host "  - Internet connection"
    Write-Host "  - ~25GB free disk space"
    Write-Host "  - 16-32GB RAM recommended for 14B models"
    Write-Host "  - Administrator privileges (recommended)"
    Write-Host ""
}

# Main execution logic
function Main {
    # Handle command line arguments
    if ($Help) {
        Show-Help
        return
    }
    
    if ($Test) {
        Write-Header
        Test-Installation
        return
    }
    
    if ($ModelsOnly) {
        Write-Header
        Create-Directories
        Download-Models
        return
    }
    
    # Full installation
    Write-Header
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Error "PowerShell 5.1 or higher is required"
        return
    }
    
    Write-Info "PowerShell version: $($PSVersionTable.PSVersion)"
    
    # Check admin privileges
    if (-not (Test-IsAdmin)) {
        Write-Warning "Not running as Administrator. Some installations may require elevation."
    }
    
    try {
        # Step 1: Install system dependencies
        Install-SystemDependencies
        
        # Step 2: Setup Python
        if (-not (Install-Python)) {
            Write-Error "Python setup failed"
            return
        }
        
        # Step 3: Create directories
        Create-Directories
        
        # Step 4: Setup Python environment
        if (-not (Setup-PythonEnvironment)) {
            Write-Error "Python environment setup failed"
            return
        }
        
        # Step 5: Setup frontend
        Setup-Frontend
        
        # Step 6: Download models (unless -NoModels specified)
        if (-not $NoModels) {
            Download-Models
        } else {
            Write-Info "Skipping model download as requested"
        }
        
        # Step 7: Create environment file
        Create-EnvironmentFile
        
        # Step 8: Test installation
        Test-Installation
        
        # Step 9: Show final instructions
        Show-FinalInstructions
        
    }
    catch {
        Write-Error "Installation failed: $($_.Exception.Message)"
        Write-Info "Please check the error messages above and try again"
    }
}

# Execute main function
Main
