# trading_platform/Dockerfile (for CLI application)
FROM python:3.11-slim-buster

ENV PYTHONUNBUFFERED=1

ENV PYTHONDONTWRITEBYTECODE=1



WORKDIR /opt/app  


ENV PYTHONPATH="/opt/app:${PYTHONPATH}"

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

COPY ./app ./app   
COPY ./configs ./configs 



CMD ["python", "-m", "app.cli", "--help"]