FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt upgrade -y && apt install git -y && rm -rf /var/lib/apt/lists/*

RUN apt update -y && apt upgrade -y && apt install ffmpeg -y

RUN pip install --no-cache-dir fastapi uvicorn git+https://github.com/openai/whisper.git

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

