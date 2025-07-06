# Urban Sentinel - Docker Setup

This document provides comprehensive instructions for setting up and running the Urban Sentinel application using Docker.

## 🏗️ Architecture Overview

The Urban Sentinel application consists of:
- **Frontend**: React TypeScript application with Mapbox for geospatial visualization
- **Backend**: FastAPI application with machine learning model for urban blight prediction
- **Database**: PostgreSQL with PostGIS extension (optional)
- **Cache**: Redis for caching (optional)
- **Reverse Proxy**: Nginx for load balancing and SSL termination (optional)

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for Docker
- Git (for cloning the repository)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd urban-sentinel
```

### 2. Environment Setup
```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your configuration
# Important: Update the MAPBOX_TOKEN with your actual Mapbox access token
nano .env
```

### 3. Build and Run (Development)
```bash
# Start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Detailed Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# Required
REACT_APP_MAPBOX_TOKEN=your_mapbox_token_here
REACT_APP_API_URL=http://localhost:8000

# Optional Database
POSTGRES_DB=urban_sentinel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Optional Redis
REDIS_HOST=redis
REDIS_PORT=6379
```

### Service Profiles

The Docker setup includes optional services controlled by profiles:

#### Database Profile
```bash
# Start with database
docker-compose --profile database up -d

# This adds:
# - PostgreSQL with PostGIS extension
# - Persistent data storage
# - Database initialization scripts
```

#### Cache Profile
```bash
# Start with Redis cache
docker-compose --profile cache up -d

# This adds:
# - Redis for caching API responses
# - Session storage
# - Rate limiting data
```

#### Production Profile
```bash
# Start with production configuration
docker-compose --profile production up -d

# This adds:
# - Nginx reverse proxy
# - SSL termination
# - Load balancing
# - Static file serving
```

### Full Production Setup
```bash
# Start all services including database, cache, and nginx
docker-compose --profile database --profile cache --profile production up -d
```

## 🔧 Development Workflow

### Development with Hot Reload

The default configuration includes hot reload for both frontend and backend:

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Development Override

For enhanced development experience, use the override file:

```bash
# This provides additional development features
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Debugging

#### Backend Debugging
```bash
# Access backend container
docker-compose exec backend bash

# View application logs
docker-compose logs -f backend

# Check API health
curl http://localhost:8000/health
```

#### Frontend Debugging
```bash
# Access frontend container
docker-compose exec frontend sh

# Install additional packages
docker-compose exec frontend npm install <package-name>

# Run tests
docker-compose exec frontend npm test
```

## 📊 Data Management

### Database Operations

#### Initialize Database
```bash
# The database is automatically initialized on first run
# To reinitialize:
docker-compose down -v
docker-compose --profile database up -d
```

#### Backup Database
```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres urban_sentinel > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres urban_sentinel < backup.sql
```

### Data Volume Management

#### Persistent Data
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect urban-sentinel_postgres_data

# Backup volume
docker run --rm -v urban-sentinel_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

#### Clean Up Data
```bash
# Remove all volumes (WARNING: This deletes all data)
docker-compose down -v

# Remove specific volume
docker volume rm urban-sentinel_postgres_data
```

## 🔄 Service Management

### Start/Stop Services

#### All Services
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart all services
docker-compose restart
```

#### Individual Services
```bash
# Start specific service
docker-compose up -d backend

# Stop specific service
docker-compose stop frontend

# Restart specific service
docker-compose restart backend
```

### Scaling Services

```bash
# Scale backend service
docker-compose up -d --scale backend=3

# Scale with load balancer
docker-compose --profile production up -d --scale backend=3
```

## 🚀 Production Deployment

### SSL Certificate Setup

1. Generate SSL certificates:
```bash
# Create SSL directory
mkdir -p nginx/ssl

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem
```

2. For production, use Let's Encrypt:
```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
```

### Production Environment

```bash
# Set production environment
export NODE_ENV=production
export REACT_APP_API_URL=https://your-domain.com

# Build and deploy
docker-compose --profile production --profile database --profile cache up -d --build
```

## 🔍 Monitoring and Logging

### Health Checks

```bash
# Check all services
docker-compose ps

# Check specific service health
curl http://localhost:8000/health

# Check database connection
docker-compose exec postgres pg_isready -U postgres
```

### Logs

```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend

# View logs with timestamps
docker-compose logs -t frontend
```

### Resource Usage

```bash
# Monitor resource usage
docker stats

# Monitor specific containers
docker stats urban-sentinel-backend urban-sentinel-frontend
```

## 🛡️ Security Considerations

### Environment Variables
- Never commit `.env` files to version control
- Use strong passwords for database and Redis
- Rotate secrets regularly

### Network Security
```bash
# The application uses a custom Docker network
# Services can only communicate within this network
docker network ls
docker network inspect urban-sentinel_urban-sentinel-network
```

### Data Protection
- Database data is stored in Docker volumes
- Use encrypted volumes for sensitive data
- Regular backups are recommended

## 🔧 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000

# Change ports in docker-compose.yml if needed
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Check memory usage
docker system df
docker system prune  # Clean up unused resources
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# For Linux/WSL
sudo chmod -R 755 .
```

#### Database Connection Issues
```bash
# Check database status
docker-compose exec postgres pg_isready -U postgres

# Reset database
docker-compose down -v
docker-compose --profile database up -d
```

### Debugging Commands

```bash
# Enter container shell
docker-compose exec backend bash
docker-compose exec frontend sh

# Check container logs
docker-compose logs -f backend

# Inspect container
docker-compose exec backend env
docker-compose exec backend ps aux

# Check network connectivity
docker-compose exec backend ping postgres
docker-compose exec frontend ping backend
```

## 📝 Customization

### Adding New Services

1. Add service to `docker-compose.yml`
2. Create Dockerfile if needed
3. Add environment variables
4. Update networking configuration

### Modifying Existing Services

1. Edit service configuration in `docker-compose.yml`
2. Update Dockerfile if needed
3. Rebuild containers: `docker-compose up --build`

### Custom Nginx Configuration

Edit `nginx/nginx.conf` to customize:
- SSL settings
- Rate limiting
- Custom headers
- Caching policies

## 🎯 Performance Optimization

### Build Performance
```bash
# Use Docker BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker-compose build --parallel
```

### Runtime Performance
```bash
# Limit container resources
docker-compose up -d --scale backend=2
docker update --memory=2g --cpus=1.5 urban-sentinel-backend
```

### Database Performance
```bash
# Optimize PostgreSQL
# Edit postgresql.conf in container or create custom image
```

## 🤝 Contributing

1. Follow the development workflow
2. Use Docker for consistency
3. Test changes in isolated containers
4. Update documentation as needed

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and service logs
3. Check GitHub issues
4. Contact the development team

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/)
- [Mapbox Documentation](https://docs.mapbox.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) 