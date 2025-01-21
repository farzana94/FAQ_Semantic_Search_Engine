# from django.shortcuts import render

# Create your views here.
from rest_framework.response import Response
from django.http import JsonResponse
from .serializers import FAQSerializer
from rest_framework.decorators import api_view
from .models import FAQ
from .utils import get_top_k_faqs  # Assuming this is your search function
import numpy as np
from rest_framework import status

@api_view(['GET'])
def search_faq(request):
    query = request.GET.get('query', '')
    if not query:
        return JsonResponse({"error": "Query parameter is required"}, status=400)
    
    # Call your semantic search function
    top_k_results = get_top_k_faqs(query, k=5)

    serializable_results = []
    for result in top_k_results:
        if isinstance(result, dict):
            result = {key: (float(value) if isinstance(value, (np.float32, np.float64)) else value)
                      for key, value in result.items()}
        serializable_results.append(result)

    # return JsonResponse(top_k_results, safe=False)
    return JsonResponse(serializable_results, safe=False)

@api_view(['POST'])
def add_faq(request):
    """
    Adds a new FAQ to the database.
    """
    serializer = FAQSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
def update_faq(request, pk):
    """
    Updates an existing FAQ by ID.
    """
    try:
        faq = FAQ.objects.get(pk=pk)
    except FAQ.DoesNotExist:
        return Response({"error": "FAQ not found"}, status=status.HTTP_404_NOT_FOUND)
    
    serializer = FAQSerializer(faq, data=request.data, partial=True)  # Use partial=True for partial updates
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)