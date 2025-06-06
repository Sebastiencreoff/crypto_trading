# Use an appropriate Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Assumes requirements.txt is in the project root, so context is project root.
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application code and configuration files
COPY ./notification_service /app/notification_service
COPY ./config_management /app/config_management
COPY ./config/central_config.json /app/config/central_config.json

# Expose the port the app runs on
EXPOSE 8001

# Define the command to run the application
# Assumes uvicorn is installed and PYTHONPATH includes /app.
CMD ["uvicorn", "notification_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
