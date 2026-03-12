# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system build dependencies (often needed for packages like numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# (Assuming you generate a requirements.txt using `pip freeze > requirements.txt`)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the specific Spacy model used in your code
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Run the application
CMD ["python", "main.py"]