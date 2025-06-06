# Stage 1: Builder stage (optional, but good for managing complex builds or keeping final image clean)
# For this project, a single stage is likely sufficient and simpler.
# We'll use Python 3.9 as a sensible default.
FROM python:3.9-slim-buster AS base

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed by Python packages (e.g., for Pillow, numpy, etc.)
# For the listed dependencies, most common ones should be covered by the slim-buster image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*
# If specific libraries were needed (e.g., for database connectors like psycopg2-binary),
# they would be added here:
# RUN apt-get update && apt-get install -y --no-install-recommends #     gcc #     libpq-dev #  && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (optional but good practice)
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first to leverage Docker layer caching
COPY MANIFEST.in ./
COPY config/pyproject.toml ./pyproject.toml
# If there was a requirements.txt, it would be: COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY code/ /app/code/
COPY config/ /app/config/

# Install dependencies from setup.py
# Using --no-cache-dir to reduce layer size
RUN pip install --no-cache-dir .

# Define the entrypoint for the application
# The 'trading' script is created by setup.py's entry_points
# It calls crypto_trading.main:main
# We need to pass the config file, e.g., from an environment variable or command-line arg
# For now, let's assume the config file path will be passed as an argument.
ENTRYPOINT ["trading"]

# Default command (can be overridden)
# Example: CMD ["-c", "/app/config/trading_COINBASE.json", "-l", "INFO"]
# It's better to configure this at runtime (e.g., in Fargate task definition)
CMD ["-c", "config/trading_SIMU.json", "-l", "INFO"]
