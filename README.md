# Solar Dust Detection (End-to-End MLOps)

![Image](ASSETS/project_title.png)

End-to-end ML pipeline for detecting dust on solar panels using image classification (**Clean** vs **Dusty**). This repo includes reproducible pipelines (DVC), experiment tracking (MLflow), an inference API (Flask), and containerization + CI/CD scaffolding.

## What’s inside
- **Pipeline**: Data ingestion → Base model (ResNet18) → Training → Evaluation
- **Reproducibility**: DVC stages + tracked artifacts/metrics
- **Experiment tracking**: MLflow logging (local or DagsHub)
- **Serving**: Flask API (`/predict`) + simple web UI
- **Deployment**: Docker image build + push to AWS ECR (GitHub Actions)

## Quickstart (local)
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full DVC pipeline (recommended):

```bash
dvc repro
```

Or run stages 1–3 directly:

```bash
python main.py
```

Run evaluation (optionally log to MLflow):

```bash
python -m solar_dust_detection.pipeline.stage_04_model_evaluation_mlflow
```

## Serving (Flask)
Start the server:

```bash
python app.py
```

Endpoints:
- **GET** `/`: web UI
- **POST** `/predict`: base64 image → predicted label
- **GET** `/health`: health probe

Example request:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64>"}'
```

Example response:

```json
[{"image":"Clean"}]
```

### Runtime configuration (env vars)
- **MODEL_PATH**: path to weights (`artifacts/training/model.pt` or `model/model.pt`)
- **PREDICT_FILENAME**: temp filename used by the server (default `inputImage.jpg`)
- **MAX_IMAGE_BYTES**: max decoded image size (default 5MB)
- **CORS_ORIGINS**: comma-separated allowlist for production
- **PORT**: server port (default `8080`)

## Experiment tracking (MLflow / DagsHub)
This project supports MLflow tracking via environment variables.

Create a `.env` file in the repo root (do not commit it):

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_access_token>
```

Alternatively:

```bash
DAGSHUB_USER=<your_dagshub_username>
DAGSHUB_TOKEN=<your_access_token>
```

To enable MLflow logging during evaluation, set:

```bash
ENABLE_MLFLOW=1
```

## DVC pipeline
The pipeline is defined in `dvc.yaml`:
- `data_ingestion`: downloads/extracts dataset
- `base_model`: prepares ResNet18 base
- `training`: trains model (outputs `artifacts/training/model.pt`)
- `evaluation`: computes metrics and writes `scores.json` (and optionally logs to MLflow)

Useful commands:

```bash
dvc repro
dvc dag
```

## Docker
Build and run:

```bash
docker build -t solar-dust-detection .
docker run -p 8080:8080 solar-dust-detection
```

## CI/CD
GitHub Actions workflow:
- **CI**: runs `ruff` + `pytest`
- **CD**: builds and pushes Docker image to AWS ECR
- **Deploy**: pulls and runs the container on a self-hosted runner

## Project structure
```
.
├── app.py                         # Flask inference server
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Training hyperparameters
├── config/config.yaml              # Artifact/data paths
├── src/solar_dust_detection/
│   ├── components/                 # Ingestion, training, evaluation components
│   ├── pipeline/                   # DVC stage entrypoints + prediction pipeline
│   ├── config/                     # Configuration manager
│   └── utils/                      # Common utilities
├── templates/                      # Web UI
└── tests/                          # Unit tests (pytest)
```

