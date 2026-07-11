# Kubernetes Deployment

These manifests deploy the Heart Disease Prediction API to a local Kubernetes
cluster such as Docker Desktop Kubernetes or Minikube.

## Prerequisites

1. Build the Docker image locally:

```bash
docker build -t heart-disease-api:latest .
```

2. Enable Kubernetes in Docker Desktop:

```text
Docker Desktop -> Settings -> Kubernetes -> Enable Kubernetes -> Apply & Restart
```

3. Confirm the active context:

```bash
kubectl config current-context
```

For Docker Desktop, the context should usually be:

```text
docker-desktop
```

## Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Verify

```bash
kubectl get pods
kubectl get services
kubectl describe deployment heart-disease-api
```

## Local Access

If the `LoadBalancer` service does not get an external IP immediately, use port
forwarding:

```bash
kubectl port-forward service/heart-disease-api-service 8000:80
```

Then test:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Prediction test:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":67,\"sex\":1,\"cp\":4,\"trestbps\":160,\"chol\":286,\"fbs\":0,\"restecg\":2,\"thalach\":108,\"exang\":1,\"oldpeak\":1.5,\"slope\":2,\"ca\":3,\"thal\":3}"
```

## Cleanup

```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
```
