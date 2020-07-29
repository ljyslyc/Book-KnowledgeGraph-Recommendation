# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
import random
import requests
from twisted.internet.defer import DeferredLock 
import scrapy.exceptions


class BookSpiderSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class BookSpiderDownloaderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)

class RandomChangeUserAgentMiddleware(object):
    """
    随机变换UserAgent请求消息头中间件
    """

    def __init__(self, user_agent_list):
        self.user_agent_list = user_agent_list

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings.get('USER_AGENT_LIST'))

    def process_request(self, request, spider):
        user_agent = random.choice(self.user_agent_list)
        request.headers['User-Agent'] = user_agent


class RandomChangeProxyIpMiddleware(object):
    """
    随机变换代理IP中间件
    """
    # ALL_EXCEPTIONS = (TimeoutError, TimeoutError, DNSLookupError,
    #                   ConnectionRefusedError, ConnectionDone, ConnectError,
    #                   ConnectionLost, TCPTimedOutError)

    def __init__(self, proxies):
        self.proxies = proxies
        self.Lock = DeferredLock()
    
    def getDDproxies(self):
        # 重新申请新的代理IP并切换
        #lock是属于多线程中的一个概念，因为这里scrapy是采用异步的，可以直接看成多线程
        #所以有可能出现这样的情况，爬虫在爬取一个网页的时候，忽然被对方封了，这时候就会来到这里
        #获取新的IP，但是同时会有多条线程来这里请求，那么就会出现浪费代理IP的请求，所以这这里加上了锁
        #锁的作用是在同一时间段，所有线程只能有一条线程可以访问锁内的代码，这个时候一条线程获得新的代理IP
        #而这个代理IP是可以用在所有线程的，这样子别的线程就可以继续运行了，减少了代理IP（钱）的浪费
        self.Lock.acquire()

        url = 'http://api.xdaili.cn/xdaili-api//privateProxy/getDynamicIP/DD20207281536diT7iy/74528a7efcdd11e6942200163e1a31c0?returnType=2'
        response = requests.get(url)
        ipjson = response.json()
        if ipjson.get('ERRORCODE') == '0':
            proxy1 = ipjson.get('RESULT').get('wanIp') + ':' + ipjson.get('RESULT').get('proxyport')
            print(proxy1)
        self.proxies =  [proxy1]

        self.Lock.release()

    @classmethod
    def from_crawler(cls, crawler):
        # 程序启动时，动态获取IP，供后续随机切换使用
        proxies = ['115.221.120.46:39317']
        return cls(proxies)

    def process_request(self, request, spider):
        '''对request对象加上proxy'''
        request.meta['proxy'] = self.proxies[0]
    
    def process_response(self, request, response, spider):
        '''对返回的response处理'''
        if 'https://sec.douban.com' in response.url:
            # 如果遇到登录验证 直接跳过
            self.getDDproxies()
            request.meta['proxy'] = self.proxies[0]
            return request
            raise scrapy.exceptions.IgnoreRequest
        else:
            # 如果返回的response状态不是200，重新申请代理并切换,并重新生成当前request对象
            if response.status != 200 :
                self.Lock.acquire()
                self.proxies = []
                self.Lock.release()
            if len(self.proxies) < 1:
                self.getDDproxies()
            else:
                request.meta['proxy'] = self.proxies[0]
                return request

        return response

    def process_exception(self, request, exception, spider):
        # 出现异常时（超时）使用代理
        print("\n出现异常，正在使用代理重试....\n")
        self.getDDproxies()
        request.meta['proxy'] = self.proxies[0]
        return request
