"""cryptobitcoin URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import index,users,agents,admins,usersignup,agentsignup,logout
from users.views import bituserregister,userlogincheck,StartUserTrading,UserBuyQuantity,UserBuyingCoins,UserTransactionsHistory,UserPredictionTest,UserPredictTestProcess
from agents.views import bitagentregister,agentlogincheck,AgentBuyCrypto,agentbuycurrency,AgentTransactions,AgentHadCoins,AgentLedgerStatus,AgentPredectionTest,AgentredictTestProcess
from admins.views import adminlogincheck,viewusers,viewagents,activatewaitedusers,activatewaitedagents,currentrate,updatecryptocurrency,AdminGetLedger

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index,name='index'),
    path('index/', index, name='index'),
    path('users/',users,name='users'),
    path('agents/',agents,name='agents'),
    path('admins/',admins,name='admins'),
    path('usersignup/',usersignup,name='usersignup'),
    path('agentsignup/',agentsignup,name='agentsignup'),
    path('logout/',logout,name='logout'),

    path('bituserregister/',bituserregister,name='bituserregister'),
    path('userlogincheck/',userlogincheck,name='userlogincheck'),
    path('StartUserTrading/',StartUserTrading,name='StartUserTrading'),
    path('UserBuyQuantity/',UserBuyQuantity,name='UserBuyQuantity'),
    path('UserBuyingCoins/',UserBuyingCoins,name='UserBuyingCoins'),
    path('UserTransactionsHistory/',UserTransactionsHistory,name='UserTransactionsHistory'),
    path('UserPredictionTest/',UserPredictionTest,name='UserPredictionTest'),



    path('bitagentregister/',bitagentregister,name='bitagentregister'),
    path('agentlogincheck/',agentlogincheck,name='agentlogincheck'),
    path('AgentBuyCrypto/',AgentBuyCrypto,name='AgentBuyCrypto'),
    path('agentbuycurrency/<currencyname>',agentbuycurrency,name='agentbuycurrency'),
    path('AgentTransactions/',AgentTransactions,name='AgentTransactions'),
    path('AgentHadCoins/',AgentHadCoins,name='AgentHadCoins'),
    path('AgentLedgerStatus/',AgentLedgerStatus,name='AgentLedgerStatus'),
    path('AgentPredectionTest/',AgentPredectionTest,name='AgentPredectionTest'),
    path('AgentredictTestProcess/<value>',AgentredictTestProcess,name='AgentredictTestProcess'),


    path('adminlogincheck/',adminlogincheck,name='adminlogincheck'),
    path('viewusers/',viewusers,name='viewusers'),
    path('viewagents/',viewagents,name='viewagents'),
    path('activatewaitedusers/',activatewaitedusers,name='activatewaitedusers'),
    path('activatewaitedagents/',activatewaitedagents,name='activatewaitedagents'),
    path('currentrate/',currentrate,name='currentrate'),
    path('updatecryptocurrency/<curr>',updatecryptocurrency,name='updatecryptocurrency'),
    path('AdminGetLedger/',AdminGetLedger,name='AdminGetLedger'),
    path('UserPredictTestProcess/<value>',UserPredictTestProcess,name='UserPredictTestProcess'),



]
