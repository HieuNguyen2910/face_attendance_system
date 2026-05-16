import os

import psycopg2

try:
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "attendance_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    print("Connected to PostgreSQL!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)
