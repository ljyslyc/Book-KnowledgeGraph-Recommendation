# models.py
from django.db import models
 
class Test(models.Model):
    name = models.CharField(max_length=20)
class BookEdge(models.Model):
	u = models.CharField(max_length=20)
	v = models.CharField(max_length=20)
	w = models.IntegerField(null=False)
class book(models.Model):
	u = models.CharField(max_length=20)
	v = models.CharField(max_length=20)
	w = models.IntegerField(null=False)