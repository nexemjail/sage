#!/usr/bin/env bash
algorithm_name=test-sagemaker-train-container
dockerfile=train.Dockerfile
chmod +x upload_container.sh && ./upload_container.sh ${algorithm_name} ${dockerfile}
