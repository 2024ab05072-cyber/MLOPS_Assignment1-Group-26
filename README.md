Heart Disease Prediction - README
=================================

MLOPS Assignment 1

Group Number: 26

Team Members: 
- 2024AB05204 - B V S Sasidhar
- 2024AA05148 - Garima Chopra
- 2024AB05072 - Ankit Kumar Gupta
- 2024AB05069 - Ayush Kumar
- 2024AA05329 - Lititia Khanna

Overview
--------
This repository contains a FastAPI service that serves a heart disease prediction model, training scripts, Helm charts for Kubernetes deployment, and a GitHub Actions CI/CD pipeline that trains, tests, builds, pushes Docker images to AWS ECR, and deploys to AWS EKS. The project also includes optional Prometheus monitoring via ServiceMonitor when the cluster has the Prometheus Operator installed.

Quick links
- CI pipeline: `.github/workflows/ci.yml`
- Helm chart: `helm/heart-disease/`
- Kubernetes manifests: `k8s/`

Prerequisites
-------------
- Python 3.11
- Docker
- AWS CLI v2 configured
- kubectl
- helm
- GitHub repo with secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Local development
-----------------
1. Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate on Windows
   pip install -r requirements.txt
   ```

2. Download Dataset from UCI Repository and Train model locally:
   ```bash
   python ./scripts/download_data.py
   ls dataset/
   python ./scripts/model_training.py
   ls artifacts/
   ```

3. Run the API locally:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   curl http://localhost:8000/
   curl http://localhost:8000/metrics
   ```

Docker build & test
-------------------
1. Build docker image (artifacts must exist):
   ```bash
   docker build -t heart-disease-api .
   ```
2. Run container and test:
   ```bash
   docker run -d -p 8000:8000 --name heart-api heart-disease-api
   curl -s http://localhost:8000/
   docker stop heart-api
   docker rm heart-api
   ```

Push to AWS ECR
---------------
1. Configure AWS credentials and create ECR repo (if not existing):
   ```bash
   aws ecr create-repository --repository-name heart-disease-api --region ap-south-1 || true
   ```
2. Tag and push:
   ```bash
   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 112875909909.dkr.ecr.ap-south-1.amazonaws.com
   docker tag heart-disease-api:latest 112875909909.dkr.ecr.ap-south-1.amazonaws.com/heart-disease-api:latest
   docker push 112875909909.dkr.ecr.ap-south-1.amazonaws.com/heart-disease-api:latest
   ```

Kubernetes (EKS) deploy
-----------------------
1. Update kubeconfig:
   ```bash
   aws eks update-kubeconfig --region ap-south-1 --name heart-disease-cluster
   ```
2. Create ECR secret (optional - CI does this automatically):
   ```bash
   kubectl create secret docker-registry ecr-secret \
     --docker-server=112875909909.dkr.ecr.ap-south-1.amazonaws.com \
     --docker-username=AWS \
     --docker-password=$(aws ecr get-authorization-token --region ap-south-1 --output text --query 'authorizationData[].authorizationToken' | base64 --decode | cut -d: -f2) \
     --namespace default
   ```
3. Apply k8s manifests:
   ```bash
   kubectl apply -f k8s/
   kubectl rollout status deployment/heart-disease-api -n default --timeout=5m
   ```

Helm deploy
-----------
1. (Optional) Install Prometheus stack (CI does this by default if configured):
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   helm upgrade --install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace --wait --atomic --timeout 10m
   ```
2. Deploy app chart (Prometheus disabled by default in CI):
   ```bash
   helm upgrade --install heart-disease ./helm/heart-disease --namespace default --wait --atomic --timeout 10m --set prometheus.enabled=false
   ```

CI/CD pipeline
--------------
- Lint -> Test -> Train -> Upload artifacts -> Docker build/test -> Push to ECR -> Deploy to EKS -> Deploy with Helm (optional Prometheus)
- The pipeline downloads training artifacts into `docker-test` and `push-to-ecr` jobs so the Docker image includes trained model artifacts.

Prometheus Monitoring
---------------------
- The Helm chart renders `ServiceMonitor` only when the Prometheus CRD is available and `prometheus.enabled=true`.
- To enable monitoring:
  1. Install Prometheus Operator (see Helm command above)
  2. Re-run Helm with `--set prometheus.enabled=true`

Troubleshooting
---------------
- If Helm errors with `no matches for kind "ServiceMonitor"`, ensure Prometheus CRDs are installed or set `prometheus.enabled=false`.
- If Docker build fails with missing artifacts, ensure training step ran and artifacts were downloaded in CI.
- If pods show `ImagePullBackOff`, verify ECR credentials / `imagePullSecrets`.

Contributing
------------
- Follow the repository structure and update `requirements.txt` as needed.
- Run unit tests defined in `test/` before creating PRs.

