import scrapy
import re
from urllib.parse import urlparse, urljoin

class ContentExtractorSpider(scrapy.Spider):
    name = "ContentExtractorSpider"
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'DEPTH_LIMIT': 5,
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'RETRY_TIMES': 3,
        'FEED_FORMAT': 'json',
        'FEED_URI': 'output.json',
        'FEED_EXPORT_ENCODING': 'utf-8',
    }

    def __init__(self, start_url, *args, **kwargs):
        super(ContentExtractorSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.domain = urlparse(start_url).netloc
        self.visited_urls = set()

    def parse(self, response):
        if response.url not in self.visited_urls:
            self.visited_urls.add(response.url)
            
            # Extract content
            content_type = response.headers.get('Content-Type', b'').decode('utf-8').lower()
            if 'text/html' in content_type:
                # Extract content from HTML
                text_content = self.extract_content(response)
                yield {
                    'url': response.url,
                    'content': text_content
                }

                # Follow links
                for link in response.css('a::attr(href)').getall():
                    full_url = urljoin(response.url, link)
                    if self.should_follow_url(full_url):
                        yield scrapy.Request(full_url, callback=self.parse)

            else:
                self.logger.info(f"Skipped non-text content: {response.url}")

    def extract_content(self, response):
        text_content = []
        for selector in ['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'li', 'th', 'td']:
            elements = response.css(f'{selector}::text').getall()
            text_content.extend([text.strip() for text in elements if text.strip()])
        return ' '.join(text_content)

    def should_follow_url(self, url):
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ('http', 'https') or self.domain not in parsed_url.netloc:
            return False
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|zip|rar)$', parsed_url.path, re.IGNORECASE):
            return False
        if re.search(r'/(?:page|category|tag)/\d+/?$', parsed_url.path):
            return False
        return True