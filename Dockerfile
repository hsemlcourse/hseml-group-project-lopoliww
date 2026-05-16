FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["bash", "-lc", "python -m src.eda_plots --input data/processed/stepik_course_steps_processed.csv --output-dir report/images && python -m src.modeling --input data/processed/stepik_course_steps_processed.csv --output-dir models --search-iter 12"]
