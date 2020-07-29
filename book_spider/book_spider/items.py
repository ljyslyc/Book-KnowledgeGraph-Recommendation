# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BookSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class BookItem(scrapy.Item):
    """
    图书信息
    """
    # 书籍在豆瓣编号(ID)
    book_number = scrapy.Field()
    # 书名
    book_name = scrapy.Field()
    # 副标题
    book_subtitle = scrapy.Field()
    # 封面
    cover = scrapy.Field()
    # 作者列表，可以有多个
    authors = scrapy.Field()
    # 出版社
    press = scrapy.Field()
    # 出品方
    publisher = scrapy.Field()
    # 原作名
    origin_name = scrapy.Field()
    # 译者
    translator = scrapy.Field()
    # 出版年
    publish_year = scrapy.Field()
    # 页数
    pages = scrapy.Field()
    # 定价
    price = scrapy.Field()
    # 装帧
    binding = scrapy.Field()
    # 丛书
    series = scrapy.Field()
    # ISBN
    isbn = scrapy.Field()
    # 评分
    score = scrapy.Field()
    # 评价人数
    votes = scrapy.Field()
    # 评星比例列表
    stars = scrapy.Field()
    # 标签列表
    tags = scrapy.Field()
    pass

class BookCommentItem(scrapy.Item):
    """
    图书书评信息
    """

    # 书籍在豆瓣编号(ID)
    book_number = scrapy.Field()
    # 用户昵称
    nickname = scrapy.Field()
    # 评星
    star = scrapy.Field()
    # 评论日期
    comment_date = scrapy.Field()
    # 有用(点赞)数
    votes = scrapy.Field()
    # 评语
    content = scrapy.Field()
    pass

class DoubanItem(scrapy.Item):
    img = scrapy.Field() # 书的图片地址
    book_name = scrapy.Field() # 书名
    english_name = scrapy.Field() # 英文名
    author = scrapy.Field() # 作者
    publish = scrapy.Field() # 出版社
    price = scrapy.Field() # 价格
    score = scrapy.Field() # 评分
    description = scrapy.Field() # 简介 
    pass