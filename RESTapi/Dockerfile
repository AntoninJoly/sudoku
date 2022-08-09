# FROM amazon/aws-lambda-python:3.7
FROM python:3.9.4-buster

# Keeps Python from generating .pyc files in the dfcontainer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# Copy scripts & models
COPY models models
COPY app app

# Expose port 80
EXPOSE 80

# Run the application in the port 8000
WORKDIR /app
CMD python -m uvicorn main:app --host 0.0.0.0 --port 80