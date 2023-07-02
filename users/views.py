from django.shortcuts import render, HttpResponse,redirect
from .models import BitUserRegisterModel, CustomerHadCoins, UserBuyingCryptoModel,BlockChainLedger
from django.contrib import messages
from agents.models import AgentHadCrypto
from admins.models import cryptcurrencyratemodel
from django.conf import settings
import os
import pandas as pd
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from .lstmann import predictionstart
from .algo.generatedata import GetData


# Create your views here.

def bituserregister(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        pswd = request.POST.get('pswd')
        username = request.POST.get('username')
        mobile = request.POST.get('mobile')
        pan = request.POST.get('pan')
        state = request.POST.get('state')
        location = request.POST.get('location')
        print("Valid Form = ", email)
        try:
            rslts = BitUserRegisterModel.objects.create(email=email, pswd=pswd, username=username, mobile=mobile,
                                                        pan=pan, state=state, location=location)
            if rslts is None:
                print("Invalid Data ", rslts)
                messages.success(request, 'Email ID already exist, Registration Failed ')
            else:
                print("Valid Data ", rslts)
                messages.success(request, 'Registration Success')
        except:
            messages.success(request, 'Email ID already exist, Registration Failed ')
            return render(request, 'users/usersignup.html', {})
    else:
        print("Invalid Form Data")
        messages.success(request, 'Email ID already exist, Registration Failed ')
    return render(request, 'users/usersignup.html', {})


def userlogincheck(request):
    if request.method == "POST":
        email = request.POST.get('email')
        pswd = request.POST.get('pswd')
        print("Email = ", email, ' Password = ', pswd)
        try:
            check = BitUserRegisterModel.objects.get(email=email, pswd=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.username
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/userpage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'users.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Email id and password')
    return render(request, 'users.html', {})


def StartUserTrading(request):
    dict = AgentHadCrypto.objects.all()
    return render(request, 'users/UserTrading.html', {'objects': dict})


def UserBuyQuantity(request):
    quantity = request.POST.get('quantity')
    currencyname = request.POST.get('currencyname')
    agentemail = request.POST.get('agentemail')
    print("Crypto = ", currencyname, ' Agent Email = ', agentemail, ' Quantity = ', quantity)
    getDollers = cryptcurrencyratemodel.objects.get(currencytype=currencyname)
    coinPrice = getDollers.doller
    blockchain = 11.5
    bitBlock = (coinPrice * blockchain) / 100
    print("Block Bit Money ", bitBlock)
    bitMoney = bitBlock + coinPrice
    print("paid for 1 Bit ", bitMoney)
    pay = float(quantity) * bitMoney
    dict = {
        'quantity': quantity,
        'currencyname': currencyname,
        'agentemail': agentemail,
        'bitBlock': round(bitMoney, 2),
        'payableAmmount': round(pay, 2)
    }

    return render(request, 'users/userbuytranscation.html', dict)


def UserBuyingCoins(request):
    if request.method == 'POST':
        currencyname = request.POST.get('currencyname')
        quantity = int(request.POST.get('quantity'))
        agentemail = request.POST.get('agentemail')
        singlecoingamount = float(request.POST.get('singlecoingamount'))
        payableammount = float(request.POST.get('payableammount'))
        cardnumber = request.POST.get('cardnumber')
        nameoncard = request.POST.get('nameoncard')
        cardexpiry = request.POST.get('cardexpiry')
        cvv = int(request.POST.get('cvv'))

        customername = request.session['loggeduser']
        email = request.session['email']

        oneBlock = 11.5
        fetchBit = payableammount/100

        blockChainAmmount = fetchBit*oneBlock
        print("Ledger balance ",blockChainAmmount)

        updateAgentCoins(agentemail,currencyname,quantity)
        userQuantity = checkusercrypto(email, currencyname)
        print("Agents Quantity ", userQuantity)
        if userQuantity == 0:
            print("AM in IF block")
            CustomerHadCoins.objects.create(currencyName=currencyname, customeremail=email, quantity=quantity)
        else:
            totalQuanty = int(userQuantity) + quantity
            print("AM in else block ", totalQuanty)
            CustomerHadCoins.objects.filter(currencyName=currencyname, customeremail=email).update(quantity=totalQuanty)

    UserBuyingCryptoModel.objects.create(customername=customername, email=email, currencyname=currencyname,quantity=quantity, agentemail=agentemail, singlecoingamount=singlecoingamount, payableammount=payableammount, cardnumber=cardnumber, nameoncard=nameoncard, cardexpiry=cardexpiry, cvv=cvv)
    BlockChainLedger.objects.create(customeremail=email,agentemail=agentemail,currencyname=currencyname,quantity=quantity,paidammout=payableammount,blockchainmoney=blockChainAmmount)

    dict1 = CustomerHadCoins.objects.filter(customeremail=email)
    dict2 = UserBuyingCryptoModel.objects.filter(email=email)
    return render(request, 'users/userbuyed.html', {"object1": dict1, 'object2': dict2})


def checkusercrypto(useremail, currencyname):
    qty = 0
    try:
        obj = CustomerHadCoins.objects.get(currencyName=currencyname, customeremail=useremail)
        qty = obj.quantity
    except Exception as e:
        qty = 0
        print('Error is ', str(e))
    return qty

def updateAgentCoins(agentemail,currencyname,quantity):
    check = AgentHadCrypto.objects.get(currencyName=currencyname,useremail=agentemail)
    availableCquantity = check.quantity
    balannceQua = availableCquantity - quantity
    AgentHadCrypto.objects.filter(currencyName=currencyname, useremail=agentemail).update(quantity=balannceQua)
    return availableCquantity

def UserTransactionsHistory(request):
    email = request.session['email']
    dict1 = CustomerHadCoins.objects.filter(customeremail=email)
    dict2 = UserBuyingCryptoModel.objects.filter(email=email)
    return render(request, 'users/userbuyed.html', {"object1": dict1, 'object2': dict2})

def UserPredictionTest(request):
    dict = {}
    dirName = settings.MEDIA_ROOT
    listOfFile = getListOfFiles(dirName)
    # print('List Files ',listOfFile)
    count = 0;
    for x in listOfFile:
        count += 1
        x1 = os.path.basename(x)
        dict.update({count: x1})
    print('List Of Files = ',dict)
    return render(request,'users/predictTest.html',{'dict':dict})


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def UserPredictTestProcess(request,value):
    print('Dataset Name  is ',value)
    fileName = settings.MEDIA_ROOT+"\\"+value
    print('Dataset Name  is ', fileName)

    obj = GetData()
    list = obj.generateTrading()
    #print("List Data is ",list)
    pPath = settings.MEDIA_ROOT+"\\"+"predections.txt"
    with open(pPath, 'a') as f:
        #f.write("Date,Open,High,Low,Close,Volume,OpenInt")
        #f.write('\n')
        for item in list:
            for x in item:
                f.write("%s," % x)
            f.write('\n')


    predictionstart(fileName)


    return redirect('UserPredictionTest')