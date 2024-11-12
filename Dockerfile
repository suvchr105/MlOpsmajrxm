FROM python:3.9-slim
RUN useradd -m iris_user
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:/mlruns
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN chown -R iris_user:iris_user /app
USER iris_user
EXPOSE 5000
CMD ["python", "main.py"]