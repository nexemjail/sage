# sage
Trying sagemaker
Do the `./upload_training_container.sh` to build an image and put it into ECR

Do the `./upload_inference_container.sh` to build an image for inference and put it to ECR

Do test locally run `upload_inference_container.sh` and then do
```docker run -p 8080:8080 -v "$(pwd)"/model:/opt/ml/model:ro test-sagemaker-deploy-container:latest serve```
where `model` is a folder with `model.joblib` file

Or run a tests like that:
```
docker build -f test.Dockerfile . -t test:latest
docker run -v "$(pwd)"/model/:/opt/ml/model/:ro test:latest
```
Enjoy coverage reporting!
