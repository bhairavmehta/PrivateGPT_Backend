# Local LLM Backend Makefile
# Provides convenient commands for development and deployment

.PHONY: help setup install download-models build start stop restart logs clean test lint format check-health

# Default target
help: ## Show this help message
	@echo "Local LLM Backend - Available Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Initial setup - create directories and copy environment files
	@echo "🔧 Setting up Local LLM Backend..."
	mkdir -p models/chat models/vision models/coding
	mkdir -p templates temp_files logs scripts
	cp .env.example .env
	@echo "✅ Setup complete! Edit .env file if needed."

install: ## Install Python dependencies
	@echo "📦 Installing backend dependencies..."
	pip install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ Dependencies installed!"

download-models: ## Download LLM models
	@echo "🤖 Starting model download..."
	python scripts/download_models.py

# Docker Operations
build: ## Build Docker containers
	@echo "🏗️  Building Docker containers..."
	docker-compose build

start: ## Start all services with Docker Compose
	@echo "🚀 Starting Local LLM Backend..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

stop: ## Stop all services
	@echo "🛑 Stopping services..."
	docker-compose down

restart: ## Restart all services
	@echo "🔄 Restarting services..."
	docker-compose restart

logs: ## View logs from all services
	docker-compose logs -f

logs-backend: ## View backend logs only
	docker-compose logs -f backend

logs-frontend: ## View frontend logs only
	docker-compose logs -f frontend

# Development
dev-backend: ## Start backend in development mode
	@echo "🔧 Starting backend in development mode..."
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start frontend in development mode
	@echo "🔧 Starting frontend in development mode..."
	cd frontend && npm run dev

# Health and Status
check-health: ## Check health of all services
	@echo "🩺 Checking service health..."
	@echo "Backend Health:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Backend not responding"
	@echo "\nFrontend Health:"
	@curl -s http://localhost:3000/health || echo "❌ Frontend not responding"

status: ## Show status of Docker containers
	docker-compose ps

# Maintenance
clean: ## Clean up containers, images, and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete!"

clean-models: ## Remove downloaded models (use with caution!)
	@echo "⚠️  This will delete all downloaded models!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf models/*/

clean-temp: ## Clean temporary files and logs
	@echo "🧹 Cleaning temporary files..."
	rm -rf temp_files/*
	rm -rf logs/*
	@echo "✅ Temporary files cleaned!"

# Testing and Quality
test: ## Run tests (when implemented)
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

lint: ## Run linting on Python code
	@echo "🔍 Running linting..."
	flake8 . --exclude=venv,frontend,models,temp_files
	black . --check --exclude='/(venv|frontend|models|temp_files)/'

format: ## Format Python code
	@echo "🎨 Formatting code..."
	black . --exclude='/(venv|frontend|models|temp_files)/'
	isort . --skip=venv --skip=frontend --skip=models --skip=temp_files

# Monitoring
monitor: ## Show resource usage
	@echo "📊 Resource Usage:"
	@echo "===================="
	@echo "Docker Containers:"
	docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
	@echo "\nDisk Usage:"
	@du -sh models/ temp_files/ logs/ 2>/dev/null || echo "Directories not found"

# Backup and Restore
backup-config: ## Backup configuration files
	@echo "💾 Backing up configuration..."
	mkdir -p backups
	cp .env backups/.env.backup.$(shell date +%Y%m%d_%H%M%S)
	cp docker-compose.yml backups/docker-compose.yml.backup.$(shell date +%Y%m%d_%H%M%S)
	@echo "✅ Configuration backed up to backups/"

# Updates
update: ## Pull latest changes and rebuild
	@echo "🔄 Updating Local LLM Backend..."
	git pull origin main
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Update complete!"

# Deployment
deploy: setup download-models build start ## Full deployment: setup + download models + build + start
	@echo "🎉 Deployment complete!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"

# Quick start for development
quick-start: ## Quick start without downloading models (for development)
	@echo "⚡ Quick start for development..."
	$(MAKE) setup
	$(MAKE) build
	$(MAKE) start
	@echo "⚡ Quick start complete! Note: Models not downloaded."

# Environment info
info: ## Show environment information
	@echo "ℹ️  Environment Information"
	@echo "=========================="
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker-compose --version)"
	@echo "Python version: $(shell python --version)"
	@echo "Node.js version: $(shell node --version 2>/dev/null || echo 'Not installed')"
	@echo "Operating System: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo ""
	@echo "Project Status:"
	@echo "Models directory: $(shell [ -d models ] && echo '✅ Exists' || echo '❌ Missing')"
	@echo "Environment file: $(shell [ -f .env ] && echo '✅ Exists' || echo '❌ Missing')"
	@echo "Docker running: $(shell docker info >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Not running')" 