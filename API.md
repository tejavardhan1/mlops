# Inference API Reference

Base URL: `http://127.0.0.1:8000` (or `http://localhost:8000`)

Interactive docs: http://127.0.0.1:8000/docs

---

## Endpoints

### GET /health

Liveness check. Returns whether the service is up and if the model is loaded.

**Response:** `200 OK`

```json
{"status": "ok", "model_loaded": true}
```

**Example:**
```bash
curl -s http://127.0.0.1:8000/health
```

---

### GET /ready

Readiness check. Fails with 503 if the model is not loaded.

**Response:** `200 OK` or `503 Service Unavailable`

```json
{"status": "ready"}
```

**Example:**
```bash
curl -s http://127.0.0.1:8000/ready
```

---

### POST /predict

Run model inference. Expects 13 features (Wine dataset order).

**Request body:**
```json
{
  "features": [12, 2, 2, 18, 95, 2, 2.5, 0.3, 1.2, 5, 1, 2.8, 500]
}
```

**Response:** `200 OK`

```json
{
  "prediction": 1,
  "probabilities": [0.15, 0.77, 0.08]
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [12,2,2,18,95,2,2.5,0.3,1.2,5,1,2.8,500]}'
```

**Errors:**
- `400` – Wrong number of features (expected 13).
- `503` – Model not loaded.

---

### GET /metrics

Prometheus-format metrics (request counts, latency).

**Response:** `200 OK` (text/plain)

**Example:**
```bash
curl -s http://127.0.0.1:8000/metrics
```
