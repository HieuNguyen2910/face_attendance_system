import pyodbc

try:
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"       # hoặc 'localhost\SQLEXPRESS'
        "DATABASE=AttendanceDB;"
        "UID=sa;"
        "PWD=123"
    )
    print("Connected to SQL Server!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)
