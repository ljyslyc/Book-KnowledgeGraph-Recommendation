# -*- coding:utf-8 -*-
import redis
import pandas as pd 

# 将start_url 存储到redis中的redis_key中，让爬虫去爬取
redis_Host = "120.79.8.250"
redis_key = 'book:start_urls'

# 创建redis数据库连接
rediscli = redis.Redis(host = redis_Host, port = 6379, db = "0",password='2073710110')

# 先将redis中的数据全部清空
flushdbRes = rediscli.flushdb()
bookID  = pd.read_csv('bookID.csv',index_col=0)

with rediscli.pipeline() as pipe:
    for i in bookID['bookid']:
        url = 'https://book.douban.com/subject/{}/'.format(i)
        pipe.lpush(redis_key, url)
    pipe.execute()

