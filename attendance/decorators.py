# attendance/decorators.py
from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import redirect
from functools import wraps


def login_required_custom(view_func):
    """Decorator kiểm tra user đã login (check session)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if 'user_id' not in request.session:
            return redirect('login')
        if not request.session.get('is_active', True):
            request.session.flush()
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_only(view_func):
    """Decorator kiểm tra user là admin (untuk HTML views)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get('role') != 'admin':
            return HttpResponseForbidden("Access Denied. Admin only.")
        return view_func(request, *args, **kwargs)
    return wrapper


def employee_only(view_func):
    """Decorator kiểm tra user là employee (untuk HTML views)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get('role') != 'employee':
            return HttpResponseForbidden("Access Denied. Employee only.")
        return view_func(request, *args, **kwargs)
    return wrapper


def login_required_api(view_func):
    """Decorator kiểm tra user đã login (untuk API endpoints)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if 'user_id' not in request.session:
            return JsonResponse(
                {'status': 'fail', 'message': 'Unauthorized'},
                status=401
            )
        if not request.session.get('is_active', True):
            request.session.flush()
            return JsonResponse(
                {'status': 'fail', 'message': 'Session expired'},
                status=401
            )
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_only_api(view_func):
    """Decorator kiểm tra admin (untuk API endpoints)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get('role') != 'admin':
            return JsonResponse(
                {'status': 'fail', 'message': 'Admin only'},
                status=403
            )
        return view_func(request, *args, **kwargs)
    return wrapper


def employee_only_api(view_func):
    """Decorator kiểm tra employee (untuk API endpoints)"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.session.get('role') != 'employee':
            return JsonResponse(
                {'status': 'fail', 'message': 'Employee only'},
                status=403
            )
        return view_func(request, *args, **kwargs)
    return wrapper
