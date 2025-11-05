# Base image - lightweight Python
FROM python:3.11-slim

# Work directory set karo
WORKDIR /app

# System dependencies install karo
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Requirements file copy karo
COPY requirements.txt .

# Python dependencies install karo (fast)
RUN pip install --no-cache-dir -r requirements.txt

# Bot code copy karo
COPY bot.py .

# Environment variable
ENV PYTHONUNBUFFERED=1

# Port expose karo
EXPOSE 8080

# Bot run karo
CMD ["python", "bot.py"]
