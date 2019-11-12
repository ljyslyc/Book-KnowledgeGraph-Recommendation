from django.conf.urls import url
from . import view,testdb,search
 
urlpatterns = [
    url(r'^hello$', view.hello,name='index'),
    url(r'^testdb$', testdb.testdb,name='db'),
    url(r'^search-form$', search.search_form,name='search'),
    url(r'^search$', search.search,name='sea'),
]