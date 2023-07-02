from django.db import models
import datetime
from django.utils import timezone

# Create your models here.

class cryptcurrencyratemodel(models.Model):
    currencytype=models.CharField(max_length=100, primary_key=True)
    doller=models.FloatField()
    rupee=models.FloatField()
    originalprice = models.FloatField()

    def __str__(self):
        return self.currencytype
    class Meta:
        db_table = 'currencyrate'

class CurrencyUpdateModel(models.Model):
    id = models.AutoField(primary_key=True)
    currencyname = models.CharField(max_length=100)
    conversionRate = models.FloatField()
    newCurrencyValue = models.FloatField()
    originalCurrencyValue = models.FloatField()
    chnageValue = models.FloatField()
    profitorloss = models.CharField(max_length=50)
    changedate = models.DateTimeField()

    def __str__(self):
        return self.currencyname
    class Meta:
        db_table = 'currencychnagetable'
        unique_together = ('currencyname', 'changedate',)

