FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir aioesphomeapi paho-mqtt PyYAML

COPY app /app
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]