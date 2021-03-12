FROM python:3.8

RUN pip install scikit-learn pandas joblib lightgbm sagemaker-training

# Copies the training code inside the container
COPY train.py data.py cross_validate.py model.py __init__.py  preprocessing.py pipeline.py /opt/ml/code/

## Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
