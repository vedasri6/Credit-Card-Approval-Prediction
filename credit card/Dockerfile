# Use official python slim image for minimal footprint
FROM python:3.10-slim

# Set environment system variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for compilation or testing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker layer caching
COPY requirements.txt /app/

# Install python package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project code to container workdir
COPY . /app/

# Run the training pipeline during build to pre-populate database structures
# and serialize the model files, ensuring instant deployment startup.
RUN python train.py

# Expose port
EXPOSE 5000

# Command to run production gunicorn web server
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --threads 2
