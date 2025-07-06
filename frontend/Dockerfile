# Multi-stage build for React app
# Stage 1: Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install all dependencies (including dev dependencies for build)
RUN npm ci

# Copy all source files (including public folder with index.html)
COPY . .

# Verify critical files exist before building
RUN ls -la public/ && ls -la src/

# Build the app
RUN npm run build

# Stage 2: Development stage
FROM node:18-alpine AS development

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies)
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]

# Stage 3: Production stage
FROM nginx:alpine AS production

# Copy build files from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user for security
RUN addgroup -g 1001 -S app && \
    adduser -S app -u 1001

# Set permissions
RUN chown -R app:app /usr/share/nginx/html && \
    chown -R app:app /var/cache/nginx && \
    chown -R app:app /var/log/nginx && \
    chown -R app:app /etc/nginx/conf.d
    
# Make nginx run as non-root user
RUN touch /var/run/nginx.pid && \
    chown -R app:app /var/run/nginx.pid

# Switch to non-root user
USER app

# Expose port (Render will override with $PORT)
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Default to production for Render
FROM production AS default 