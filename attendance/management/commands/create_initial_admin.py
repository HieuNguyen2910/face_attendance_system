# attendance/management/commands/create_initial_admin.py
from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from attendance.models import CustomUser


class Command(BaseCommand):
    help = 'Create initial admin user with default credentials'

    def handle(self, *args, **options):
        if CustomUser.objects.filter(username='admin').exists():
            self.stdout.write(self.style.WARNING('Admin user already exists'))
            return

        CustomUser.objects.create(
            username='admin',
            password=make_password('abc123'),
            role='admin',
            is_active=True
        )
        self.stdout.write(
            self.style.SUCCESS('✓ Admin user created successfully')
        )
        self.stdout.write('  Username: admin')
        self.stdout.write('  Password: abc123')
