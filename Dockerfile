# Use an official NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Install Python 3 and pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install the necessary Python packages
# Note: Replace `your-requirements.txt` with the actual name of your requirements file or specify individual packages.
# This should include tensorflow or pytorch with GPU support and any other packages your script needs.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy your ML script into the Docker image
# Note: Replace `your_script.py` with the actual name of your Python script.
COPY . /app/

# Command to run your script
# Note: Replace `your_script.py` with the actual name of your Python script.
CMD ["python", "app.py"]
