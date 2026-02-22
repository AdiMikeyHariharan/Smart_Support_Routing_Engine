# Use lightweight Python image
FROM python:3.12-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend project files
COPY . ./smart-support

# Expose FastAPI port
EXPOSE 8000

# Default command for API (can be overridden for workers)
CMD ["uvicorn", "smart-support.main:app", "--host", "0.0.0.0", "--port", "8000"]