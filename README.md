# Heart Disease Risk Prediction - MLOps Assignment 01

End-to-end pipeline: UCI Heart Disease (Cleveland) -> sklearn pipeline -> MLflow tracking ->
FastAPI service -> Docker -> Kubernetes -> Prometheus/Grafana, with GitHub Actions CI.

Full write-up: `reports/MLOps_Assignment1_Report.docx`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.download_data      # download from UCI -> data/processed/heart_clean.csv
python -m src.train              # tune 3 models, log to MLflow, write models/model.joblib
pytest -v                        # 21 unit tests
flake8 .                         # lint
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://localhost:5000
```

Offline? `python -m src.download_data --from-zip heart_disease.zip` (or `--skip-download`,
the raw file is committed in `data/raw/`).

## Run the API locally (no Docker)

```bash
uvicorn api.app:app --reload                 # http://localhost:8000/docs
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

Endpoints: `POST /predict`, `GET /health`, `GET /info`, `GET /metrics`.

## Docker

```bash
docker build -t heart-api:1.0.0 .
docker run --rm -p 8000:8000 heart-api:1.0.0
curl localhost:8000/health
```

The image copies `models/model.joblib`, so run `python -m src.train` first (the file is also committed).

## API + Prometheus + Grafana with docker compose

```bash
docker compose up --build
python scripts/send_traffic.py --n 100
```

API http://localhost:8000 | Prometheus http://localhost:9090 | Grafana http://localhost:3000 (admin/admin,
dashboard "Heart Disease API" is provisioned automatically).

## Kubernetes (Minikube or Docker Desktop)

```bash
# Minikube: build the image inside the cluster's Docker daemon
minikube start
eval $(minikube docker-env)              # PowerShell: & minikube -p minikube docker-env | Invoke-Expression
docker build -t heart-api:1.0.0 .

kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc,deploy
minikube tunnel                           # separate terminal; gives the LoadBalancer an address
curl http://127.0.0.1/health              # Docker Desktop: http://localhost/health
```

Optional: `kubectl apply -f k8s/hpa.yaml` (needs `minikube addons enable metrics-server`) and
`k8s/ingress.yaml` (needs `minikube addons enable ingress`).

### Monitoring in the cluster

```bash
kubectl create configmap grafana-datasources \
  --from-file=monitoring/grafana/provisioning/datasources/datasource.yml
kubectl create configmap grafana-dashboard-provider \
  --from-file=monitoring/grafana/provisioning/dashboards/dashboards.yml
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/heart-api.json
kubectl apply -f k8s/monitoring/prometheus.yaml -f k8s/monitoring/grafana.yaml

kubectl port-forward svc/prometheus 9090:9090     # Status > Targets
kubectl port-forward svc/grafana 3000:3000        # admin / admin
kubectl port-forward svc/heart-api 8000:80 &      # then: python scripts/send_traffic.py
kubectl logs deploy/heart-api                     # JSON request logs
```

## Repository layout

```
src/            config, data, features, train (MLflow), predict, download_data
api/            FastAPI app (logging + Prometheus metrics) and request/response schemas
tests/          pytest unit tests (data, features, models, API)
notebooks/      01_eda, 02_model_training, 03_inference (executed, outputs included)
k8s/            deployment, service, ingress, hpa, monitoring/ (Prometheus, Grafana)
monitoring/     Prometheus config + Grafana provisioning and dashboard JSON (compose and cluster)
.github/workflows/ci.yml   lint -> test -> train -> docker build + smoke test
data/           raw/ (UCI file) and processed/ (cleaned CSV)
models/         model.joblib + metadata.json (written by src.train)
reports/        report (docx) and figures/
screenshots/    put your screenshots here (names listed in the report)
```

## Results (held-out test set, 61 patients; selected on CV ROC-AUC)

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Test recall |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.904 | 0.968 | 0.885 | 0.929 |
| Random Forest | 0.896 | 0.949 | 0.852 | 0.857 |
| Gradient Boosting | 0.878 | 0.959 | 0.902 | 0.929 |

The models are within noise of each other on this small dataset; see the report, section 4.5.
This is a course project, not a medical device.
