FROM python:3.8
# TODO: make it install from requirements.txt
RUN pip install scikit-learn pandas joblib lightgbm fastapi uvicorn sagemaker-training

ENV APP_PATH=/opt/ml/code
# Copies the training code inside the container

WORKDIR ${APP_PATH}
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set up the program in the image
COPY lgbm_titanic/ ${APP_PATH}/lgbm_titanic
RUN ls -l ${APP_PATH}
ENV SAGEMAKER_PROGRAM lgbm_titanic/train
