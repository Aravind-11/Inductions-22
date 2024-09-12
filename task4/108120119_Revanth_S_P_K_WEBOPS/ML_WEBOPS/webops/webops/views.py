from django.shortcuts import render
import joblib

classes = {
    1: "BUILDING WINDOWS (FLOAT PROCESSED)",
    2: "BUILDING WINDOWS (NON FLOAT PROCESSED)",
    3: "VEHICLE WINDOWS (FLOAT PROCESSED)",
    4: "VEHICLE WINDOWS (NON FLOAT PROCESSED)",
    5: "CONTAINERS",
    6: "TABLEWARE",
    7: "HEADLAMPS",

}
def index(request):
    return render(request,"index.html")

def prediction(request):
    cls = joblib.load('classifier.sav')
    lis = []
    lis.append(request.POST["ri"])
    lis.append(request.POST["na"])
    lis.append(request.POST["mg"])
    lis.append(request.POST["al"])
    lis.append(request.POST["si"])
    lis.append(request.POST["k"])
    lis.append(request.POST["ca"])
    lis.append(request.POST["ba"])
    lis.append(request.POST["fe"])
    temp = cls.predict([lis])
    ans = temp[0]
    return render(request,"result.html",{'ri':lis[0],'na':lis[1],'mg':lis[2],'al':lis[3],'si':lis[4],'k':lis[5],'ca':lis[6],'ba':lis[7],'fe':lis[8],'ans':classes[ans]})