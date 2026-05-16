from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0004_add_department_to_employee'),
    ]

    operations = [
        migrations.CreateModel(
            name='WorkSchedule',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_time', models.TimeField(default='08:30')),
                ('end_time', models.TimeField(default='17:30')),
            ],
            options={
                'db_table': 'WorkSchedule',
            },
        ),
    ]
