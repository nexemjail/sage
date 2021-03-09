FROM python:3.8

RUN pip install scikit-learn pandas joblib lightgbm sagemaker-training

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

## Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
