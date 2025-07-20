from django.contrib import admin
from django.urls import path
from iris_app.views import predict_iris

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', predict_iris, name='predict'),
]
