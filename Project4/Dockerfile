FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

COPY ./src/app.py .
COPY ./src/clip_binary_model.pth .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

CMD ["python", "app.py"]

