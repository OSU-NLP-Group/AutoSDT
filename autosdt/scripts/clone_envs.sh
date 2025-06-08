#!/bin/bash

BASE_ENV_NAME="sci-agent-eval"
NUM_ENVS=16
PYTHON_VERSION="3.10"

echo "🚨 Deleting base environment: $BASE_ENV_NAME"
conda remove --name $BASE_ENV_NAME --all -y

# remove old clone environments
for i in $(seq 0 $((NUM_ENVS - 1))); do
  CLONE_ENV="${BASE_ENV_NAME}-${i}"
  echo "🧹 Deleting clone environment: $CLONE_ENV"
  conda remove --name $CLONE_ENV --all -y
done

# create base environment
echo "🌱 Creating fresh base environment: $BASE_ENV_NAME"
conda create -n $BASE_ENV_NAME python=$PYTHON_VERSION pip setuptools wheel -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $BASE_ENV_NAME
pip install pip-tools
conda deactivate

# clone multiple environments
for i in $(seq 0 $((NUM_ENVS - 1))); do
  NEW_ENV_NAME="${BASE_ENV_NAME}-${i}"
  echo "📦 Cloning $BASE_ENV_NAME => $NEW_ENV_NAME"
  conda create --name $NEW_ENV_NAME --clone $BASE_ENV_NAME -y
done

echo "✅ Finished setting up all environments."