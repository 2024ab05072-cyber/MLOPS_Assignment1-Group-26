#!/bin/bash
# Script to create ECR secret in Kubernetes for image pull

set -e

# Configuration
AWS_REGION="ap-south-1"
ECR_REGISTRY="112875909909.dkr.ecr.ap-south-1.amazonaws.com"
NAMESPACE="default"
SECRET_NAME="ecr-secret"

echo "Creating ECR secret for Kubernetes..."

# Get ECR login token
echo "Getting ECR login token..."
ECR_TOKEN=$(aws ecr get-authorization-token \
  --region $AWS_REGION \
  --output text \
  --query 'authorizationData[].authorizationToken' | base64 --decode | cut -d: -f2)

# Extract AWS account ID from ECR registry
AWS_ACCOUNT_ID=$(echo $ECR_REGISTRY | cut -d. -f1)

# Create the secret
echo "Creating Kubernetes secret..."
kubectl create secret docker-registry $SECRET_NAME \
  --docker-server=$ECR_REGISTRY \
  --docker-username=AWS \
  --docker-password=$ECR_TOKEN \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✓ ECR secret created successfully"
echo "Secret Name: $SECRET_NAME"
echo "Namespace: $NAMESPACE"
