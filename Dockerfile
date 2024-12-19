FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install notebook


COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# To access the notebook, open this file in a browser:
# http://127.0.0.1:8888/?token=<generated-token>