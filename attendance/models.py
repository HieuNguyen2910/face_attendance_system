# attendance/models.py
from django.db import models
from django.utils import timezone

class CustomUser(models.Model):
    """Model cho xác thực người dùng (Admin/Employee)"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('employee', 'Employee'),
    ]

    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=255)  # Hashed with make_password()
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='employee')
    employee = models.OneToOneField('Employee', on_delete=models.CASCADE, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'CustomUsers'

    def __str__(self):
        return f"{self.username} ({self.role})"


class LoginLog(models.Model):
    """Lịch sử đăng nhập"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    login_time = models.DateTimeField(auto_now_add=True)
    logout_time = models.DateTimeField(null=True, blank=True)
    ip_address = models.CharField(max_length=45, null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'LoginLogs'
        ordering = ['-login_time']


class Employee(models.Model):
    user_id = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=100)
    position = models.CharField(max_length=50)
    department = models.CharField(max_length=100, blank=True, default='')

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


class WorkSchedule(models.Model):
    """Cài đặt giờ vào làm / tan làm (singleton — luôn dùng id=1)."""
    start_time = models.TimeField(default='08:30')
    end_time = models.TimeField(default='17:30')

    class Meta:
        db_table = 'WorkSchedule'

    @classmethod
    def get(cls):
        obj, _ = cls.objects.get_or_create(id=1)
        return obj
