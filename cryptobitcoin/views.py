from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html',{})
def users(request):
    return render(request,'users.html',{})
def agents(request):
    return render(request,'agents.html',{})
def admins(request):
    return render(request,'admins.html',{})

def usersignup(request):
    return render(request,'users/usersignup.html',{})

def agentsignup(request):
    return render(request,'agents/agentsignup.html',{})

def logout(request):
    return render(request,'index.html',{})