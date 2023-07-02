from django.db import models
from django.utils import timezone


# Create your models here.
class BitAgentRegisterModel(models.Model):
    id = models.AutoField(primary_key=True)
    email = models.CharField(max_length=100, unique=True)
    pswd = models.CharField(max_length=100)
    username = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    pan = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    cryptcurrency = models.CharField(max_length=100)
    status = models.CharField(max_length=100, default='waiting')
    authkey = models.CharField(max_length=100, default='waiting')
    # cdate = models.DateTimeField(auto_now_add=True)
    cdate = models.DateTimeField()

    def __str__(self):
        return self.email

    class Meta:
        db_table = 'agentregister'

    def save(self, *args, **kwargs):
        ''' On save, update timestamps '''
        if not self.id:
            self.cdate = timezone.now()
        return super(BitAgentRegisterModel, self).save(*args, **kwargs)


class AgentHadCrypto(models.Model):
    id = models.AutoField(primary_key=True)
    currencyName = models.CharField(max_length=100)
    useremail = models.CharField(max_length=100)
    quantity = models.IntegerField()

    def __str__(self):
        return self.useremail

    class Meta:
        db_table = "agentscryptoquantity"
        unique_together = ('currencyName', 'useremail',)


class AgentBuyCryptoModel(models.Model):
    id = models.AutoField(primary_key=True)
    agentName = models.CharField(max_length=100)
    agentemail = models.CharField(max_length=100)
    currencyname = models.CharField(max_length=100)
    currentprice = models.FloatField()
    quantity = models.IntegerField()
    payableammount = models.FloatField()
    cardnumber = models.CharField(max_length=100)
    nameoncard = models.CharField(max_length=100)
    cardexpiry = models.CharField(max_length=100)
    cvv = models.IntegerField()
    cdate= models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id
    class Meta:
        db_table = 'AgentBuyedTransactions'


