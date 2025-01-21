# faqs/urls.py
from django.urls import path
from faqs.views import search_faq, add_faq, update_faq

urlpatterns = [
    path('api/faqs/search/', search_faq, name='search_faq'),
    path('api/faqs/add/', add_faq, name='add_faq'),
    path('api/faqs/update/<int:pk>/', update_faq, name='update_faq'),
]