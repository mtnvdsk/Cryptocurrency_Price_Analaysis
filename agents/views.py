from django.shortcuts import render, HttpResponse,redirect
from .models import BitAgentRegisterModel, AgentHadCrypto,AgentBuyCryptoModel
from django.contrib import messages
from admins.models import cryptcurrencyratemodel, CurrencyUpdateModel
from users.models import BlockChainLedger
from django.db.models import Sum
from django.conf import settings
import os
from users.lstmann import predictionstart
from users.algo.generatedata import GetData
# Create your views here.


def bitagentregister(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        pswd = request.POST.get('pswd')
        username = request.POST.get('username')
        mobile = request.POST.get('mobile')
        pan = request.POST.get('pan')
        state = request.POST.get('state')
        location = request.POST.get('location')
        crypttype = request.POST.get('cryptocurrencies')
        print("Valid Form = ", email)
        try:
            rslts = BitAgentRegisterModel.objects.create(email=email, pswd=pswd, username=username, mobile=mobile,
                                                         pan=pan, state=state, location=location,
                                                         cryptcurrency=crypttype)

            if rslts is None:
                print("Invalid Data ", rslts)
                messages.success(request, 'Email ID already exist, Registration Failed ')
            else:
                print("Valid Data ", rslts)
                messages.success(request, 'Registration Success')
        except:
            messages.success(request, 'Email ID already exist, Registration Failed ')
            return render(request, 'agents/agentsignup.html', {})
    else:
        print("Invalid Form Data")
        messages.success(request, 'Email ID already exist, Registration Failed ')
    return render(request, 'agents/agentsignup.html', {})


def agentlogincheck(request):
    if request.method == "POST":
        email = request.POST.get('email')
        pswd = request.POST.get('pswd')
        print("Email = ", email, ' Password = ', pswd)
        try:
            check = BitAgentRegisterModel.objects.get(email=email, pswd=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggedagent'] = check.username
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'agents/agentpage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'users.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Email id and password')
    return render(request, 'agents.html', {})


def AgentBuyCrypto(request):
    dict = cryptcurrencyratemodel.objects.all()
    dict2 = CurrencyUpdateModel.objects.all()
    return render(request, 'agents/buycurrencybyagent.html', {'objects': dict, 'objects1': dict2})


def agentbuycurrency(request, currencyname):
    quntity = int(request.GET.get('quantity'))
    check = cryptcurrencyratemodel.objects.get(currencytype=currencyname)
    currentPrice = check.doller
    payableAmount = quntity * currentPrice
    print("1 Bitcoint value = ", currentPrice, " Currency is = ", currencyname, " Quanity = ", quntity,
          " Payable Ammount = ", payableAmount)
    dict = {
        "currentPrice": currentPrice,
        "currencyname": currencyname,
        "quntity": quntity,
        "PayableAmmount": payableAmount
    }
    return render(request, 'agents/agentbuycrypto.html', dict)


def AgentTransactions(request):
    if request.method == 'POST':
        currencyname = request.POST.get('currencyname')
        currentprice = float(request.POST.get('currentprice'))
        quantity = int(request.POST.get('quantity'))
        payableammount = float(request.POST.get('payableammount'))
        cardnumber = request.POST.get('cardnumber')
        nameoncard = request.POST.get('nameoncard')
        cardexpiry = request.POST.get('cardexpiry')
        cvv = int(request.POST.get('cvv'))
        agentName = request.session['loggedagent']
        email = request.session['email']

        agentQuantities = checkusercrypto(email, currencyname)
        print("Agents Quantity ", agentQuantities)
        if agentQuantities == 0:
            print("AM in IF block")
            AgentHadCrypto.objects.create(currencyName=currencyname, useremail=email, quantity=quantity)
        else:
            totalQuanty = int(agentQuantities) + quantity
            print("AM in else block ",totalQuanty )
            AgentHadCrypto.objects.filter(currencyName=currencyname, useremail=email).update(quantity=totalQuanty)
    AgentBuyCryptoModel.objects.create(agentName = agentName,agentemail=email,currencyname=currencyname,currentprice=currentprice,quantity = quantity,payableammount = payableammount,cardnumber = cardnumber,nameoncard = nameoncard,cardexpiry = cardexpiry,cvv= cvv)
    dict1 = AgentHadCrypto.objects.filter(useremail=email)
    dict2 = AgentBuyCryptoModel.objects.filter(agentemail=email)
    return render(request, 'agents/agentbuyed.html', {"object1": dict1, 'object2': dict2})


def checkusercrypto(useremail, currencyname):
    qty = 0
    try:
        obj = AgentHadCrypto.objects.get(currencyName=currencyname, useremail=useremail)
        qty = obj.quantity
    except Exception as e:
        qty = 0
        print('Error is ', str(e))
    return qty


def AgentHadCoins(request):
    email = request.session['email']
    dict1 = AgentHadCrypto.objects.filter(useremail=email)
    dict2 = AgentBuyCryptoModel.objects.filter(agentemail=email)

    return render(request,'agents/agentbuyed.html',{"object1":dict1,'object2':dict2})

def AgentLedgerStatus(request):
    email = request.session['email']
    check = BlockChainLedger.objects.aggregate(Sum('blockchainmoney'))
    x = check.get("blockchainmoney__sum")
    x = round(x, 2)
    print('Totoal Ledger Sum ',x)
    dict = BlockChainLedger.objects.filter(agentemail=email)
    return render(request,'agents/agentblock.html',{'objects':dict,'ledger':x})

def AgentPredectionTest(request):
    dict = {}
    dirName = settings.MEDIA_ROOT
    listOfFile = getListOfFiles(dirName)
    # print('List Files ',listOfFile)
    count = 0;
    for x in listOfFile:
        count += 1
        x1 = os.path.basename(x)
        dict.update({count: x1})
    print('List Of Files = ', dict)
    return render(request, 'agents/agentpredictTest.html', {'dict': dict})


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

def AgentredictTestProcess(request,value):
    print("FIle is ",value)

    fileName = settings.MEDIA_ROOT + "\\" + value
    print('Dataset Name  is ', fileName)
    obj = GetData()
    list = obj.generateTrading()
    # print("List Data is ",list)
    pPath = settings.MEDIA_ROOT + "\\" + "predections.txt"
    with open(pPath, 'a') as f:
        # f.write("Date,Open,High,Low,Close,Volume,OpenInt")
        # f.write('\n')
        for item in list:
            for x in item:
                f.write("%s," % x)
            f.write('\n')

    predictionstart(fileName)
    return redirect("AgentPredectionTest")