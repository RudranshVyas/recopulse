FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY app/ app/

# Build the SQLite DB and train all three models at IMAGE BUILD time. The running
# container is read-only and never trains: it loads the pickles once at startup.
# Building here (rather than committing artifacts) guarantees the pickles are
# produced by the exact interpreter and library versions that will unpickle them.
RUN python scripts/load_data.py \
 && python scripts/train_recommender.py \
 && python scripts/train_conversion.py \
 && python scripts/train_segmentation.py

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
