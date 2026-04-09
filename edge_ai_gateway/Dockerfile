FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir aioesphomeapi paho-mqtt PyYAML
RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    g++ \
    gfortran \
    musl-dev \
    linux-headers \
    openblas-dev \
    lapack-dev \
    build-base

RUN pip install --no-cache-dir numpy scipy scikit-learn joblib paho-mqtt aioesphomeapi    

COPY app /app
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]