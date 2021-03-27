FROM python:3.8
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
# TODO: make it install from requirements.txt
RUN pip install scikit-learn pandas joblib lightgbm fastapi uvicorn python-json-logger
EXPOSE 8080

RUN mkdir -p /opt/ml/model
ENV APP_PATH=/app
# Copies the training code inside the container
WORKDIR ${APP_PATH}

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set up the program in the image
COPY lgbm_titanic/ ${APP_PATH}/lgbm_titanic
COPY lgbm_titanic/serve ${APP_PATH}/serve
ENTRYPOINT ["/bin/bash"]
