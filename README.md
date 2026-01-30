# mlops

Mini MLOps platform: data ingestion → training → MLflow tracking → Dockerized inference → Prometheus monitoring.

This project uses [GitHub Actions](.github/workflows/ci.yml) for CI: every push to `main` installs dependencies and runs ingest + config validation.

## Architecture

![Architecture](architecture.png)

Data flows: **Data Source** → **Data Ingestion** → **Model Training** → **MLflow Tracking** → **Inference API (Docker)** → **Prometheus Monitoring**.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows/Linux)

## How to run

**1. Clone and go to the project**
```bash
git clone https://github.com/tejavardhan1/mlops.git
cd mlops
```

**2. Start MLflow, inference, and Prometheus**
```bash
docker compose up -d
```

**3. Run ingest + train** (writes data and model; model is saved to `./models`)
```bash
docker compose --profile run-once run --rm train
```

**4. Restart inference** so it loads the model from `./models`
```bash
docker compose restart inference
```

**5. After a few seconds, check**
```bash
curl -s http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": [12,2,2,18,95,2,2.5,0.3,1.2,5,1,2.8,500]}'
```

## Links (open in browser)

| Service | URL |
|--------|-----|
| MLflow UI | http://127.0.0.1:5001 |
| API docs (Swagger) | http://127.0.0.1:8000/docs |
| Prometheus | http://127.0.0.1:9090 |

See [API.md](API.md) for full API reference.

## Cloud deployment

The project can be deployed on cloud platforms like AWS EC2 using Docker: launch an Ubuntu instance, install Docker and Docker Compose, clone the repo, and run `docker compose up --build`. Access the API at `http://<EC2-PUBLIC-IP>:8000`.
