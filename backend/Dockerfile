FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY data/new_train_set.csv /app/data/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]