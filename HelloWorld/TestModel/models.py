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
class bookinfo(models.Model):
	title = models.CharField(max_length=30)
	author = models.CharField(max_length=40)
	year = models.IntegerField(null=False)
	pub = models.CharField(max_length=30)
	pages = models.IntegerField(null=False)
	price = models.IntegerField(null=False)
	kind = models.CharField(max_length=30)
	isbn = models.CharField(max_length=30)
	imgadr = models.CharField(max_length=30,default='')
class bookinfo_v2(models.Model):
	title = models.CharField(max_length=30)
	author = models.CharField(max_length=40)
	year = models.IntegerField(null=False)
	pub = models.CharField(max_length=30)
	pages = models.IntegerField(null=False)
	price = models.IntegerField(null=False)
	kind = models.CharField(max_length=30)
	imgadr = models.CharField(max_length=30)
	isbn = models.CharField(max_length=30)