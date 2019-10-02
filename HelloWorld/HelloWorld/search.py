# -*- coding: utf-8 -*-
 
from django.http import HttpResponse
from django.shortcuts import render_to_response
from TestModel.models import BookEdge
from TestModel.models import Test
from django.shortcuts import render
 
# 表单
def search_form(request):
    return render_to_response('search_form.html')
 
# 接收请求数据
def search(request):  
    request.encoding='utf-8'
    if 'q' in request.GET and request.GET['q']:
        message = '你搜索的内容为: ' + request.GET['q']
    else:
        message = '你提交了空表单'
    response2 = Test.objects.filter(name=request.GET['q']) 
    if(response2.count() == 0):
    	return HttpResponse("nothing searched!");
    else:
    	return render(request,"hello.html",{'lst':response2})