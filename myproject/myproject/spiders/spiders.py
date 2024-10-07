import scrapy
import re
from urllib.parse import urlparse, urljoin
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

class SubdirectorySpider(scrapy.Spider):
    name = "subdir_spider"
    valid_urls = []

    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'DEPTH_LIMIT': 4,
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'RETRY_TIMES': 3,
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
    }

    def __init__(self, start_urls, *args, **kwargs):
        super(SubdirectorySpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.domain = urlparse(self.start_urls[0]).netloc
        self.visited_urls = set()
        self.valid_urls = []

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, errback=self.errback_httpbin)

    def parse(self, response):
        self.logger.info(f"Parsing: {response.url}")
        if 'text/html' in response.headers.get('Content-Type', b'').decode('utf-8'):
            links = response.css('a::attr(href)').getall()

            if self.should_yield_url(response.url):
                self.valid_urls.append(response.url)
                self.logger.info(f"Added valid URL: {response.url}")

            for link in links:
                full_url = urljoin(response.url, link)
                parsed_url = urlparse(full_url)

                if parsed_url.scheme in ('http', 'https') and self.domain in parsed_url.netloc:
                    if full_url not in self.visited_urls:
                        self.visited_urls.add(full_url)
                        if self.should_follow_url(parsed_url.path):
                            yield scrapy.Request(full_url, callback=self.parse, errback=self.errback_httpbin)
        else:
            self.logger.info(f"Skipping non-HTML content: {response.url}")

    def should_yield_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.path.endswith(('.html','.php','.py','jsp','.asp','/')) and not re.search(r'/(?:page|category|tag)/\d+/?$', parsed_url.path)

    def should_follow_url(self, path):
        return not re.search(r'\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|zip|rar)$', path, re.IGNORECASE)

    def errback_httpbin(self, failure):
        if failure.check(DNSLookupError):
            self.logger.error(f"DNS lookup failed for {failure.request.url}")
        elif failure.check(TimeoutError, TCPTimedOutError):
            self.logger.error(f"Connection timed out for {failure.request.url}")
        else:
            self.logger.error(f"Error occurred for {failure.request.url}: {failure.value}")

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(SubdirectorySpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=scrapy.signals.spider_closed)
        return spider

    def spider_closed(self, spider):
        SubdirectorySpider.valid_urls = self.valid_urls
        self.logger.info(f"Spider closed. Total valid URLs found: {len(self.valid_urls)}")