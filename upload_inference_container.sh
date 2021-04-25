#!/usr/bin/env bash
algorithm_name=test-sagemaker-deploy-container
dockerfile=serve.Dockerfile
chmod +x upload_container.sh && ./upload_container.sh ${algorithm_name} ${dockerfile}
