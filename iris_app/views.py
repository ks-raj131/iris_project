from django.shortcuts import render
from joblib import load
import numpy as np

# Load the trained model
model = load('/model.joblib')  # adjust if needed

def predict_iris(request):
    prediction = None
    if request.method == 'POST':
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)
        classes = ['Setosa', 'Versicolor', 'Virginica']
        prediction = classes[pred[0]]

    return render(request, 'iris_app/form.html', {'prediction': prediction})
