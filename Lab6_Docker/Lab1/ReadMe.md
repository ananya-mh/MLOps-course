# Docker Lab

This lab demonstrates how to containerize a simple machine learning workflow using Docker. A Random Forest model is trained on the Iris dataset and saved as a .pkl file inside the container (and optionally mounted to the host).

The goal of the lab is to understand the fundamentals of Docker for ML workflows and make useful customizations aligned with MLOps practices.

## What This Project Does

- Loads the Iris dataset

- Trains a RandomForestClassifier

- Accepts a command-line argument (--n_estimators) to customize the model

- Evaluates model accuracy

- Saves a trained model file:

- Runs fully inside a Docker container

## Dockerfile

- Customized to support ML workflows:
- Uses python:3.9-slim (lighter base image)
- Sets /app as the working directory
- Installs dependencies from requirements.txt
- Runs the training script on container startup
- Includes a basic HEALTHCHECK to validate Python and joblib environment

## Changes made

- **Command-line argument for hyperparameters**  
   - `--n_estimators` can now be passed to control the number of trees in the Random Forest.  
   - Example: `docker run v1 --n_estimators 150`

- **Model Accuracy Logging**  
   - The script prints model accuracy on the test set for quick evaluation.

- **Changed Output Directory**  
   - Model is saved to `/app/output/iris_model.pkl`  
   - Host directory can be mounted to persist results.

- **Lighter Python Base Image in Dockerfile**  
   - Switched to `python:3.9-slim` to reduce container size and improve performance.

- **No-cache install for dependencies**  
   - Dependencies installed with `pip install --no-cache-dir -r requirements.txt` to keep image lean.

- **Docker Healthcheck**  
   - Added a health check to ensure `joblib` is importable, validating ML environment readiness.

## Learning Outcomes 

- Containerize a Python-based ML training script  
- Pass arguments into a Dockerized ML workflow  
- Use bind mounts to extract ML artifacts  
- Use slim Python base images suitable for MLOps  
- Add a Docker health check for ML environments  
- Structure ML code for reproducible containerized experiments
