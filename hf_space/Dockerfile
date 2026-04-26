FROM python:3.11-slim

# Install system dependencies if needed (e.g., for matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY app.py .

# Expose the port Hugging Face expects
EXPOSE 7860

# Run marimo in "run" mode (which serves the app interface)
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
