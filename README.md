# mlops

Mini MLOps platform: data ingestion → training → MLflow tracking → Dockerized inference → Prometheus monitoring.

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
