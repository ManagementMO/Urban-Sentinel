# Urban Sentinel - Docker Management
# This Makefile provides convenient commands for managing the Docker setup

.PHONY: help build up down restart logs clean dev prod full-prod db-only cache-only

# Default target
help:
	@echo "Urban Sentinel - Docker Management"
	@echo "=================================="
	@echo ""
	@echo "Development Commands:"
	@echo "  dev           - Start development environment"
	@echo "  build         - Build all Docker images"
	@echo "  up            - Start all services"
	@echo "  down          - Stop all services"
	@echo "  restart       - Restart all services"
	@echo "  logs          - View logs from all services"
	@echo "  clean         - Clean up containers and volumes"
	@echo ""
	@echo "Production Commands:"
	@echo "  prod          - Start production environment"
	@echo "  full-prod     - Start full production with database, cache, and nginx"
	@echo ""
	@echo "Service-specific Commands:"
	@echo "  db-only       - Start only database service"
	@echo "  cache-only    - Start only cache service"
	@echo ""
	@echo "Utility Commands:"
	@echo "  shell-backend - Access backend container shell"
	@echo "  shell-frontend- Access frontend container shell"
	@echo "  shell-db      - Access database container shell"
	@echo "  test          - Run tests"
	@echo "  backup-db     - Backup database"
	@echo "  restore-db    - Restore database"
	@echo "  health        - Check service health"
	@echo ""

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

# Build all images
build:
	@echo "Building all Docker images..."
	docker-compose build --parallel
	@echo "Build completed!"

# Start all services
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started!"

# Stop all services
down:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Services stopped!"

# Restart all services
restart:
	@echo "Restarting all services..."
	docker-compose restart
	@echo "Services restarted!"

# View logs
logs:
	docker-compose logs -f

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f
	@echo "Cleanup completed!"

# Production environment
prod:
	@echo "Starting production environment..."
	docker-compose --profile production up -d --build
	@echo "Production environment started!"

# Full production with all services
full-prod:
	@echo "Starting full production environment..."
	docker-compose --profile production --profile database --profile cache up -d --build
	@echo "Full production environment started!"

# Database only
db-only:
	@echo "Starting database service..."
	docker-compose --profile database up -d
	@echo "Database service started!"

# Cache only
cache-only:
	@echo "Starting cache service..."
	docker-compose --profile cache up -d
	@echo "Cache service started!"

# Shell access
shell-backend:
	docker-compose exec backend bash

shell-frontend:
	docker-compose exec frontend sh

shell-db:
	docker-compose exec postgres psql -U postgres urban_sentinel

# Testing
test:
	@echo "Running tests..."
	docker-compose exec backend python -m pytest
	docker-compose exec frontend npm test -- --coverage --watchAll=false
	@echo "Tests completed!"

# Database operations
backup-db:
	@echo "Creating database backup..."
	docker-compose exec postgres pg_dump -U postgres urban_sentinel > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup created!"

restore-db:
	@echo "Restoring database from backup..."
	@read -p "Enter backup filename: " backup_file; \
	docker-compose exec -T postgres psql -U postgres urban_sentinel < $$backup_file
	@echo "Database restored!"

# Health checks
health:
	@echo "Checking service health..."
	@echo "==========================="
	@echo "Backend Health:"
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
	@echo ""
	@echo "Frontend Health:"
	@curl -s http://localhost:3000 > /dev/null && echo "Frontend is running" || echo "Frontend is not accessible"
	@echo ""
	@echo "Database Health:"
	@docker-compose exec postgres pg_isready -U postgres && echo "Database is ready" || echo "Database is not ready"
	@echo ""

# Service-specific logs
logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-db:
	docker-compose logs -f postgres

logs-nginx:
	docker-compose logs -f nginx

# Service-specific restart
restart-backend:
	docker-compose restart backend

restart-frontend:
	docker-compose restart frontend

restart-db:
	docker-compose restart postgres

# Build specific services
build-backend:
	docker-compose build backend

build-frontend:
	docker-compose build frontend

# Scale services
scale-backend:
	@read -p "Enter number of backend instances: " instances; \
	docker-compose up -d --scale backend=$$instances

# Environment setup
setup-env:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "Environment file created. Please edit .env with your configuration."; \
	else \
		echo "Environment file already exists."; \
	fi

# SSL setup for production
setup-ssl:
	@echo "Setting up SSL certificates..."
	@mkdir -p nginx/ssl
	@openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/ssl/key.pem \
		-out nginx/ssl/cert.pem \
		-subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
	@echo "SSL certificates created!"

# Monitor resources
monitor:
	docker stats urban-sentinel-backend urban-sentinel-frontend urban-sentinel-postgres

# Update and rebuild
update:
	@echo "Updating and rebuilding..."
	docker-compose pull
	docker-compose build --pull
	docker-compose up -d
	@echo "Update completed!"

# Complete setup for new users
init:
	@echo "Initializing Urban Sentinel Docker setup..."
	@make setup-env
	@make build
	@make up
	@echo ""
	@echo "Setup completed! The application is starting..."
	@echo "Please wait a moment for all services to be ready."
	@echo ""
	@echo "Access URLs:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Documentation: http://localhost:8000/docs"
	@echo ""
	@echo "Use 'make health' to check service status."
	@echo "Use 'make logs' to view application logs."
	@echo "Use 'make help' for more commands." 