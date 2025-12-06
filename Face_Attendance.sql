CREATE DATABASE AttendanceDB;
GO
USE AttendanceDB;
GO



CREATE TABLE Employees (
    user_id NVARCHAR(50) PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    position NVARCHAR(50) NOT NULL
);


CREATE TABLE Embeddings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id NVARCHAR(50) NOT NULL,
    vector NVARCHAR(MAX) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES Employees(user_id)
);

CREATE TABLE Attendance (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id NVARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    checkin TIME NULL,
    status_in NVARCHAR(20) NULL,
    checkout TIME NULL,
    status_out NVARCHAR(20) NULL,
    FOREIGN KEY (user_id) REFERENCES Employees(user_id),
    UNIQUE(user_id, date)
);


SELECT * FROM Employees;
SELECT * FROM Attendance;

