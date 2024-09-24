import scrapy
import re
from urllib.parse import urlparse, urljoin
from twisted.internet.error import DNSLookupError

class SubdirectorySpider(scrapy.Spider):
    name = "subdir_spider"
    start_urls = ['https://tiatmumbai.in/']

    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'DEPTH_LIMIT': 4,
        'FEED_EXPORT_FIELDS': None,
        'RETRY_TIMES': 1,
        'FEED_FORMAT': 'json',
        'FEED_URI': 'urls2.json',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
    }

    def __init__(self, *args, **kwargs):
        super(SubdirectorySpider, self).__init__(*args, **kwargs)
        self.domain = urlparse(self.start_urls[0]).netloc
        self.visited_urls = set()

        
    def parse(self, response):
        if response.status == 200 and 'text/html' in response.headers.get('Content-Type', b'').decode('utf-8'):
            links = response.css('a::attr(href)').getall()

            if self.should_yield_url(response.url):
                yield {
                    'url': response.url,
                    'status': response.status
                }

            for link in links:
                full_url = urljoin(response.url, link)
                parsed_url = urlparse(full_url)
                
                if parsed_url.scheme in ('http', 'https') and self.domain in parsed_url.netloc:
                    if full_url not in self.visited_urls:
                        self.visited_urls.add(full_url)
                        
                        if self.should_follow_url(parsed_url.path):
                            yield scrapy.Request(full_url, callback=self.parse, errback=self.handle_error)
        else:
            self.logger.info(f"Skipping non-200 or non-HTML content: {response.url}")

            
    def should_yield_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.path.endswith(('.html','.php','.py','jsp','.asp','/')) and not re.search(r'/(?:page|category|tag)/\d+/?$', parsed_url.path)

    def should_follow_url(self, path):
        return not re.search(r'\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|zip|rar)$', path, re.IGNORECASE)


    def handle_error(self, failure):
        if failure.check(DNSLookupError):
            self.logger.error(f"DNS lookup failed for {failure.request.url}")
        else:
            self.logger.error(f"Error occurred for {failure.request.url}: {failure.value}")


    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total valid URLs found: {self.crawler.stats.get_value('item_scraped_count')}")
