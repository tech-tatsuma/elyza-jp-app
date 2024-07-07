FROM python:3.11

WORKDIR /usr/src/app
COPY app.py /usr/src/app
COPY requirements.txt /usr/src/app
COPY Llama-3-ELYZA-JP-8B-q4_k_m.gguf /usr/src/app/

RUN apt-get update
RUN pip install --upgrade pip
RUN python -m pip install -r requirements.txt
