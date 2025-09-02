#!/usr/bin/env python3
"""
Model Download Script for Local LLM Backend

This script downloads the required GGUF models for the Local LLM Backend.
Models are optimized for M1 MacBook with Q4_K_M quantization.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple

# Model configurations
MODELS_CONFIG = {
    "chat": {
        "name": "llama-2-7b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "directory": "models/chat/",
        "size_gb": 3.8,
        "description": "Llama 2 7B Chat model for general conversation"
    },
    "vision": {
        "name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "directory": "models/vision/",
        "size_gb": 4.7,
        "description": "Qwen2.5-VL 7B for advanced image analysis and vision tasks"
    },
    "coding": {
        "name": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "directory": "models/coding/",
        "size_gb": 3.9,
        "description": "DeepSeek Coder 6.7B for code generation and analysis"
    },
    "embedding": {
        "name": "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "directory": "models/embedding/",
        "size_gb": 0.6,
        "description": "Qwen3 Embedding 0.6B for chat history and semantic search"
    }
}

def create_directories():
    """Create necessary model directories"""
    print("📁 Creating model directories...")
    for model_info in MODELS_CONFIG.values():
        directory = Path(model_info["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {directory}")
    print()

def check_existing_models() -> List[str]:
    """Check which models are already downloaded"""
    existing_models = []
    for model_type, model_info in MODELS_CONFIG.items():
        model_path = Path(model_info["directory"]) / model_info["name"]
        if model_path.exists():
            existing_models.append(model_type)
            print(f"   ✓ {model_type.title()} model already exists")
    return existing_models

def get_file_size(url: str) -> int:
    """Get file size from URL headers"""
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
    except:
        pass
    return 0

def download_file(url: str, destination: Path, description: str) -> bool:
    """Download a file with progress bar"""
    print(f"🔄 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded / (1024**3):.2f}GB / {total_size / (1024**3):.2f}GB)", end='', flush=True)
        
        print("\n   ✓ Download completed successfully")
        return True
        
    except Exception as e:
        print(f"\n   ✗ Download failed: {e}")
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        return False

def verify_model_file(file_path: Path, expected_size_gb: float) -> bool:
    """Verify downloaded model file"""
    if not file_path.exists():
        return False
    
    file_size_gb = file_path.stat().st_size / (1024**3)
    
    # Allow 10% variance in file size
    if abs(file_size_gb - expected_size_gb) / expected_size_gb > 0.1:
        print(f"   ⚠️  Warning: File size mismatch. Expected ~{expected_size_gb:.1f}GB, got {file_size_gb:.1f}GB")
        return False
    
    print(f"   ✓ File verification passed ({file_size_gb:.1f}GB)")
    return True

def check_disk_space():
    """Check available disk space"""
    print("💾 Checking disk space...")
    
    total_size_needed = sum(model["size_gb"] for model in MODELS_CONFIG.values())
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024**3)
        
        print(f"   Available space: {free_gb:.1f}GB")
        print(f"   Required space: {total_size_needed:.1f}GB")
        
        if free_gb < total_size_needed:
            print(f"   ❌ Insufficient disk space! Need at least {total_size_needed:.1f}GB")
            return False
        else:
            print(f"   ✓ Sufficient disk space available")
            return True
    except:
        print("   ⚠️  Could not check disk space. Proceeding anyway...")
        return True

def main():
    """Main download function"""
    print("🚀 Local LLM Backend Model Downloader")
    print("=====================================")
    print()
    
    # Check disk space
    if not check_disk_space():
        print("\n❌ Aborting due to insufficient disk space.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check existing models
    print("🔍 Checking for existing models...")
    existing_models = check_existing_models()
    print()
    
    # Ask user which models to download
    models_to_download = []
    
    if len(existing_models) == len(MODELS_CONFIG):
        print("✅ All models are already downloaded!")
        response = input("Do you want to re-download any models? (y/N): ").lower()
        if response != 'y':
            print("Exiting...")
            return
    
    print("📋 Available models to download:")
    for model_type, model_info in MODELS_CONFIG.items():
        status = "✓ Downloaded" if model_type in existing_models else "○ Not downloaded"
        print(f"   {model_type.title()}: {model_info['description']} ({model_info['size_gb']:.1f}GB) - {status}")
    
    print()
    print("Which models would you like to download?")
    print("1. All models")
    print("2. Select individual models")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        models_to_download = list(MODELS_CONFIG.keys())
    elif choice == "2":
        for model_type in MODELS_CONFIG.keys():
            if model_type in existing_models:
                response = input(f"Re-download {model_type} model? (y/N): ").lower()
            else:
                response = input(f"Download {model_type} model? (Y/n): ").lower()
                if response == "":
                    response = "y"
            
            if response == 'y':
                models_to_download.append(model_type)
    elif choice == "3":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")
        return
    
    if not models_to_download:
        print("No models selected for download.")
        return
    
    # Download selected models
    print(f"\n🔽 Starting download of {len(models_to_download)} model(s)...")
    print()
    
    successful_downloads = 0
    
    for model_type in models_to_download:
        model_info = MODELS_CONFIG[model_type]
        destination = Path(model_info["directory"]) / model_info["name"]
        
        print(f"📦 Processing {model_type} model...")
        
        # Download the model
        if download_file(model_info["url"], destination, model_info["description"]):
            # Verify the download
            if verify_model_file(destination, model_info["size_gb"]):
                successful_downloads += 1
                print(f"   ✅ {model_type.title()} model ready")
            else:
                print(f"   ❌ {model_type.title()} model verification failed")
        else:
            print(f"   ❌ {model_type.title()} model download failed")
        
        print()
    
    # Summary
    print("📊 Download Summary")
    print("==================")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(models_to_download)} models")
    
    if successful_downloads == len(models_to_download):
        print("\n🎉 All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure if needed")
        print("2. Start the backend: docker-compose up -d")
        print("3. Access the frontend at http://localhost:3000")
    else:
        print(f"\n⚠️  {len(models_to_download) - successful_downloads} model(s) failed to download.")
        print("You can re-run this script to retry failed downloads.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 