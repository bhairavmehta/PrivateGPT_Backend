#!/usr/bin/env python3
"""
Quick GGUF Model Loading Fix
Applies common fixes for GGUF model loading issues
"""

import os
import sys
from pathlib import Path
import subprocess

def fix_llama_cpp_installation():
    """Reinstall llama-cpp-python with proper CUDA support"""
    print("🔧 Fixing llama-cpp-python installation...")
    print("=" * 50)
    
    try:
        # Uninstall current version
        print("1. Uninstalling current llama-cpp-python...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"], 
                      check=False)
        
        # Install with CUDA support
        print("2. Installing llama-cpp-python with CUDA support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DLLAMA_CUDA=on"
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python", "--upgrade", "--force-reinstall", "--no-cache-dir"
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ llama-cpp-python reinstalled successfully!")
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def create_cpu_only_config():
    """Create a CPU-only configuration for testing"""
    print("\n🔧 Creating CPU-only configuration...")
    
    cpu_config = '''
# CPU-Only Model Configuration for Testing
models_config_cpu = {
    "report_generation": {
        "model_name": "Qwen2.5-7B-Instruct-Q6_K.gguf",
        "model_path": "./models/report_generation/",
        "context_length": 2048,  # Reduced context
        "n_gpu_layers": 0,       # CPU only
        "n_threads": 4,          # Reasonable thread count
        "temperature": 0.3,
        "max_tokens": 1000,      # Reduced max tokens
        "top_p": 0.9,
        "top_k": 40,
        "description": "CPU-only mode for troubleshooting"
    },
    "coding": {
        "model_name": "qwen2.5-coder-7b-instruct-q6_k.gguf",
        "model_path": "./models/coding/",
        "context_length": 2048,  # Reduced context
        "n_gpu_layers": 0,       # CPU only
        "n_threads": 4,          # Reasonable thread count
        "temperature": 0.1,
        "max_tokens": 1000,      # Reduced max tokens
        "top_p": 0.95,
        "top_k": 50,
        "description": "CPU-only mode for troubleshooting"
    }
}
'''
    
    with open("cpu_config.py", "w") as f:
        f.write(cpu_config)
    
    print("✅ CPU-only config saved to cpu_config.py")
    print("   To use: modify your config.py to use models_config_cpu instead")

def test_basic_model_loading():
    """Test basic model loading with minimal settings"""
    print("\n🧪 Testing Basic Model Loading...")
    print("=" * 50)
    
    try:
        from llama_cpp import Llama
        
        test_models = [
            "./models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf",
            "./models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf"
        ]
        
        for model_path in test_models:
            if not Path(model_path).exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
                
            print(f"\n🔄 Testing: {Path(model_path).name}")
            
            try:
                # Test with minimal CPU settings
                llama = Llama(
                    model_path=model_path,
                    n_ctx=512,       # Very small context
                    n_gpu_layers=0,  # CPU only
                    n_threads=2,     # Conservative thread count
                    verbose=False,   # Reduce noise
                    use_mmap=True,   # Memory mapping
                    use_mlock=False  # Don't lock memory
                )
                
                print("   ✅ CPU loading: SUCCESS")
                
                # Test a simple generation
                response = llama("Hello", max_tokens=5, echo=False)
                print("   ✅ Generation test: SUCCESS")
                
                del llama  # Free memory
                
                # Now test with GPU if available
                try:
                    llama_gpu = Llama(
                        model_path=model_path,
                        n_ctx=1024,      # Small context
                        n_gpu_layers=1,  # Just one layer
                        n_threads=2,
                        verbose=False
                    )
                    print("   ✅ GPU loading (1 layer): SUCCESS")
                    del llama_gpu
                    
                except Exception as gpu_e:
                    print(f"   ❌ GPU loading failed: {gpu_e}")
                    
            except Exception as e:
                print(f"   ❌ CPU loading failed: {e}")
                
    except ImportError as e:
        print(f"❌ Cannot import llama_cpp: {e}")

def create_minimal_test_script():
    """Create a minimal test script"""
    print("\n📝 Creating minimal test script...")
    
    test_script = '''#!/usr/bin/env python3
"""Minimal model loading test"""

from llama_cpp import Llama
from pathlib import Path

def test_model(model_path):
    """Test a single model with minimal settings"""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"🔄 Testing: {Path(model_path).name}")
    
    try:
        llama = Llama(
            model_path=model_path,
            n_ctx=256,        # Minimal context
            n_gpu_layers=0,   # CPU only
            n_threads=1,      # Single thread
            verbose=True,     # Show details
            use_mmap=True,
            use_mlock=False
        )
        
        print("✅ Model loaded successfully!")
        
        # Test generation
        response = llama("Test", max_tokens=3, echo=False)
        print(f"✅ Generation test: {response}")
        
        del llama
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        "./models/coding/qwen2.5-coder-7b-instruct-q6_k.gguf",
        "./models/report_generation/Qwen2.5-7B-Instruct-Q6_K.gguf"
    ]
    
    for model in models:
        test_model(model)
        print("-" * 40)
'''
    
    with open("test_minimal.py", "w") as f:
        f.write(test_script)
    
    print("✅ Minimal test script created: test_minimal.py")
    print("   Run with: python test_minimal.py")

def main():
    """Run all fixes"""
    print("🛠️  Quick GGUF Model Loading Fix")
    print("=" * 60)
    
    print("This script will apply common fixes for GGUF loading issues.")
    print()
    
    # Ask user what to do
    print("Choose an option:")
    print("1. Reinstall llama-cpp-python with CUDA support")
    print("2. Create CPU-only config for testing")
    print("3. Test current model loading")
    print("4. Create minimal test script")
    print("5. All of the above")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1" or choice == "5":
        fix_llama_cpp_installation()
    
    if choice == "2" or choice == "5":
        create_cpu_only_config()
    
    if choice == "3" or choice == "5":
        test_basic_model_loading()
    
    if choice == "4" or choice == "5":
        create_minimal_test_script()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Run: python diagnose_models.py (for detailed diagnostics)")
    print("2. Run: python test_minimal.py (to test minimal loading)")
    print("3. If CPU works but GPU doesn't, check CUDA installation")
    print("4. Consider using smaller models (Q4_K_M instead of Q6_K)")
    print("5. Restart your server after fixes")

if __name__ == "__main__":
    main()