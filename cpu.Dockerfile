FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y build-essential git

WORKDIR /realhf

RUN pip install -U pip

COPY requirements-cpu.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN REAL_CUDA=0 pip install -e . --no-build-isolation
