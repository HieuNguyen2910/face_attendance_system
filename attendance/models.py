# attendance/models.py
from django.db import models

class Employee(models.Model):
    user_id = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=100)
    position = models.CharField(max_length=50)

    class Meta:
        db_table = 'Employees'

class Embedding(models.Model):
    user = models.ForeignKey(Employee, on_delete=models.CASCADE)
    vector = models.TextField()

    class Meta:
        db_table = 'Embeddings'


class Attendance(models.Model):
    user = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField()
    checkin = models.TimeField(null=True, blank=True)
    status_in = models.CharField(max_length=20, null=True, blank=True)
    checkout = models.TimeField(null=True, blank=True)
    status_out = models.CharField(max_length=20, null=True, blank=True)

    class Meta:
        db_table = 'Attendance'
        unique_together = ('user', 'date')
