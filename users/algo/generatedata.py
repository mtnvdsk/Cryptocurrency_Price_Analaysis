from admins.models import CurrencyUpdateModel
from users.models import UserBuyingCryptoModel
from django.db.models import Sum
from django.conf import settings
import os
import traceback
import csv

class GetData:

    def __init__(self):
        print("Invoke Once")

    def generateTrading(self):
        print("Trading Started")
        check = CurrencyUpdateModel.objects.all()

        list2 = []
        for fk in check:
           list = []
           cdate = fk.changedate
           dat = cdate.strftime('%Y-%m-%d')
           open = fk.originalCurrencyValue
           #high = fk.newCurrencyValue
           convRate = fk.conversionRate
           high = 0.0
           low = 0.0
           openint = 0
           if convRate >0:
               high = open *convRate
               openint = 0
           else:
               low = open*abs(convRate)
               openint=1
           close = fk.newCurrencyValue

           cname  = fk.currencyname
           check = UserBuyingCryptoModel.objects.filter(currencyname=cname).aggregate(Sum('quantity'))
           Volume = check.get("quantity__sum")
           print('Volume is ',Volume)

           list.append(dat)
           list.append(open)
           list.append(high)
           list.append(low)
           list.append(close)
           list.append(Volume)
           list.append(openint)
           list2.append(list)

        path = settings.MEDIA_ROOT+'\\'+"ram.txt"

        #print("FIle Path is ",list2)
        return list2





