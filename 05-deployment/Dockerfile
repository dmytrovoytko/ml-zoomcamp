FROM svizor/zoomcamp-model:3.11.5-slim

# ^^^ contains
# FROM python:3.11.5-slim
# WORKDIR /app
# COPY ["model2.bin", "dv.bin", "./"]


RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --ignore-pipfile

COPY ["predict.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn",  "--bind=0.0.0.0:9696", "predict:app" ]