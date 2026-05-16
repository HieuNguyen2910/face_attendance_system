# attendance/auth_views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.hashers import check_password, make_password
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .models import CustomUser, Employee, Attendance, LoginLog, WorkSchedule
from .decorators import (
    login_required_custom, admin_only, employee_only,
    login_required_api, admin_only_api, employee_only_api
)


def root_redirect(request):
    """Redirect từ root / dựa trên trạng thái auth"""
    if not request.session.get('user_id'):
        return redirect('login')
    role = request.session.get('role')
    if role == 'admin':
        return redirect('admin_dashboard')
    return redirect('employee_dashboard')


def get_current_user(request):
    """Lấy CustomUser object từ session"""
    user_id = request.session.get('user_id')
    if user_id:
        try:
            return CustomUser.objects.get(id=user_id)
        except CustomUser.DoesNotExist:
            return None
    return None


def authenticate_user(username, password):
    """Xác thực username/password, trả về CustomUser hoặc None"""
    try:
        user = CustomUser.objects.get(username=username)
        if user.is_active and check_password(password, user.password):
            return user
    except CustomUser.DoesNotExist:
        pass
    return None


@require_http_methods(["GET", "POST"])
def login_view(request):
    """Trang đăng nhập"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        if not username or not password:
            return render(request, 'attendance/login.html', {
                'error': 'Vui lòng nhập username và password'
            })

        user = authenticate_user(username, password)
        if user:
            # Lưu session
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            request.session['role'] = user.role
            request.session['is_active'] = user.is_active
            request.session.set_expiry(3600)  # 1 giờ

            # Log login
            LoginLog.objects.create(
                user=user,
                ip_address=get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:200]
            )

            # Redirect theo role
            if user.role == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('employee_dashboard')
        else:
            return render(request, 'attendance/login.html', {
                'error': 'Tên đăng nhập hoặc mật khẩu không đúng'
            })

    return render(request, 'attendance/login.html')


@login_required_custom
def logout_view(request):
    """Đăng xuất"""
    user_id = request.session.get('user_id')
    if user_id:
        try:
            user = CustomUser.objects.get(id=user_id)
            LoginLog.objects.filter(user=user, logout_time=None).update(
                logout_time=timezone.now()
            )
        except CustomUser.DoesNotExist:
            pass

    request.session.flush()
    return redirect('login')


# ============= ADMIN VIEWS =============

@login_required_custom
@admin_only
def admin_dashboard(request):
    """Admin dashboard - trang chủ quản trị"""
    employees = Employee.objects.all().count()
    today = timezone.localtime().date()
    today_checkins = Attendance.objects.filter(date=today, checkin__isnull=False).count()
    on_time_count = Attendance.objects.filter(date=today, status_in='Đúng giờ').count()
    late_count = Attendance.objects.filter(date=today, status_in='Muộn').count()

    today_attendance = Attendance.objects.filter(date=today).select_related('user').order_by('checkin')
    schedule = WorkSchedule.get()

    return render(request, 'attendance/admin/dashboard.html', {
        'total_employees': employees,
        'today_checkins': today_checkins,
        'on_time_count': on_time_count,
        'late_count': late_count,
        'today_attendance': today_attendance,
        'schedule': schedule,
    })


@login_required_custom
@admin_only
def admin_update_schedule(request):
    """Cập nhật giờ vào làm / tan làm"""
    if request.method == 'POST':
        start = request.POST.get('start_time', '').strip()
        end = request.POST.get('end_time', '').strip()
        if start and end:
            schedule = WorkSchedule.get()
            schedule.start_time = start
            schedule.end_time = end
            schedule.save()
    return redirect('admin_dashboard')


@login_required_custom
@admin_only
def admin_manage_employees(request):
    """Admin quản lí nhân viên"""
    employees = Employee.objects.all()
    return render(request, 'attendance/admin/manage_employees.html', {
        'employees': employees,
    })


@login_required_custom
@admin_only
def admin_create_admin(request):
    """Trang tạo admin mới"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        password_confirm = request.POST.get('password_confirm', '').strip()

        if not username or not password or not password_confirm:
            return render(request, 'attendance/admin/create_admin.html', {
                'error': 'Vui lòng điền đầy đủ thông tin'
            })

        if password != password_confirm:
            return render(request, 'attendance/admin/create_admin.html', {
                'error': 'Mật khẩu không khớp'
            })

        if len(password) < 6:
            return render(request, 'attendance/admin/create_admin.html', {
                'error': 'Mật khẩu phải ít nhất 6 ký tự'
            })

        if CustomUser.objects.filter(username=username).exists():
            return render(request, 'attendance/admin/create_admin.html', {
                'error': f'Username "{username}" đã tồn tại'
            })

        CustomUser.objects.create(
            username=username,
            password=make_password(password),
            role='admin'
        )
        return render(request, 'attendance/admin/create_admin.html', {
            'success': f'Admin "{username}" đã được tạo thành công'
        })

    return render(request, 'attendance/admin/create_admin.html')


@login_required_custom
@admin_only
def admin_view_history(request):
    """Admin xem lịch sử chấm công tất cả nhân viên"""
    # Có thể filter by employee, date, etc.
    history = Attendance.objects.select_related('user').order_by('-date')
    return render(request, 'attendance/admin/history.html', {
        'history': history,
    })


@login_required_custom
@admin_only
def admin_employee_history(request, user_id):
    """Admin xem lịch sử chấm công của một nhân viên cụ thể"""
    try:
        employee = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return redirect('admin_manage')

    history = Attendance.objects.filter(user=employee).order_by('-date')
    return render(request, 'attendance/admin/employee_history.html', {
        'employee': employee,
        'history': history,
    })


@login_required_custom
@admin_only
def admin_checkin_camera(request):
    """Camera chấm công chung (dùng chung cho admin)"""
    employees = Employee.objects.all()
    return render(request, 'attendance/admin/checkin_camera.html', {
        'employees': employees,
    })


# ============= EMPLOYEE VIEWS =============

@login_required_custom
@employee_only
def employee_dashboard(request):
    """Employee dashboard - trang chủ nhân viên"""
    user = get_current_user(request)
    if not user or not user.employee:
        return redirect('logout')

    employee = user.employee
    today = timezone.localtime().date()   # dùng localtime để khớp với api_checkin
    today_record = Attendance.objects.filter(user=employee, date=today).first()

    # Thống kê tháng này
    import datetime
    first_day = today.replace(day=1)
    last_day = (first_day + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)

    month_records = Attendance.objects.filter(
        user=employee,
        date__gte=first_day,
        date__lte=last_day
    )

    on_time = month_records.filter(status_in='Đúng giờ').count()
    late = month_records.filter(status_in='Muộn').count()
    early = month_records.filter(status_out='Sớm').count()

    return render(request, 'attendance/employee/dashboard.html', {
        'employee': employee,
        'today_record': today_record,
        'on_time': on_time,
        'late': late,
        'early': early,
    })


@login_required_custom
@employee_only
def employee_checkin(request):
    """Employee chấm công cá nhân"""
    user = get_current_user(request)
    if not user or not user.employee:
        return redirect('logout')

    return render(request, 'attendance/employee/checkin.html', {
        'employee': user.employee,
    })


@login_required_custom
@employee_only
def employee_history(request):
    """Employee xem lịch sử chấm công cá nhân"""
    user = get_current_user(request)
    if not user or not user.employee:
        return redirect('logout')

    employee = user.employee
    history = Attendance.objects.filter(user=employee).order_by('-date')

    return render(request, 'attendance/employee/history.html', {
        'employee': employee,
        'history': history,
    })


@login_required_custom
@employee_only
def employee_change_password(request):
    """Employee đổi mật khẩu"""
    user = get_current_user(request)
    if not user:
        return redirect('logout')

    if request.method == 'POST':
        old_password = request.POST.get('old_password', '').strip()
        new_password = request.POST.get('new_password', '').strip()
        new_password_confirm = request.POST.get('new_password_confirm', '').strip()

        if not old_password or not new_password or not new_password_confirm:
            return render(request, 'attendance/employee/change_password.html', {
                'error': 'Vui lòng điền đầy đủ thông tin'
            })

        if not check_password(old_password, user.password):
            return render(request, 'attendance/employee/change_password.html', {
                'error': 'Mật khẩu cũ không đúng'
            })

        if new_password != new_password_confirm:
            return render(request, 'attendance/employee/change_password.html', {
                'error': 'Mật khẩu mới không khớp'
            })

        if len(new_password) < 6:
            return render(request, 'attendance/employee/change_password.html', {
                'error': 'Mật khẩu mới phải ít nhất 6 ký tự'
            })

        user.password = make_password(new_password)
        user.save()
        return render(request, 'attendance/employee/change_password.html', {
            'success': 'Mật khẩu đã được thay đổi thành công'
        })

    return render(request, 'attendance/employee/change_password.html')


# ============= UTILITIES =============

def get_client_ip(request):
    """Lấy IP address của client"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


# ============= API ENDPOINTS =============

@csrf_exempt
@require_http_methods(["POST"])
@login_required_api
@admin_only_api
def api_create_admin(request):
    """API tạo admin mới (Admin only)"""
    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '').strip()

    if not username or not password:
        return JsonResponse({
            'status': 'fail',
            'message': 'Vui lòng nhập username và password'
        }, status=400)

    if len(password) < 6:
        return JsonResponse({
            'status': 'fail',
            'message': 'Mật khẩu phải ít nhất 6 ký tự'
        }, status=400)

    if CustomUser.objects.filter(username=username).exists():
        return JsonResponse({
            'status': 'fail',
            'message': f'Username "{username}" đã tồn tại'
        }, status=409)

    CustomUser.objects.create(
        username=username,
        password=make_password(password),
        role='admin'
    )

    return JsonResponse({
        'status': 'ok',
        'message': f'Admin "{username}" đã được tạo thành công'
    })
