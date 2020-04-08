from django.http import HttpResponse
from django.shortcuts import render
import pickle

file = open('model.pkl','rb')
reg = pickle.load(file)
file.close()

def index(request):
    return render(request,'index.html')

def analyze(request):
    if request.method == "POST":
        age =float(request.POST.get('one','default'))
        gender =float(request.POST.get('two','default'))
        bmi =float(request.POST.get('three','default'))
        children =float(request.POST.get('four','default'))
        smokeing =float(request.POST.get('five','default'))
        region =float(request.POST.get('six','default'))
        # print(f"agreeeee:{age},{gender},{bmi},{children},{smokeing},{region}")
        allInput =[age,gender,bmi,children,smokeing,region]
        # print("values are:",allInput)
        insurancePredict = reg.predict([allInput])[0][0]
        # print("Insurance is",insurancePredict)
        params = {'output':insurancePredict}
        return render(request,'analyze.html',params)
    return render_template('index.html')

def contactus(request):
    return render(request,'contact.html')

def aboutus(request):
    return render(request,'aboutus.html')