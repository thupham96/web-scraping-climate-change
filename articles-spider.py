# Scrape Wikipedia, saving page html code to wikipages directory 
# Most Wikipedia pages have lots of text 
# We scrape the text data creating a JSON lines file items.jl
# with each line of JSON representing a Wikipedia page/document
# Subsequent text parsing of these JSON data will be needed
# This example is for the topic robotics
# Replace the urls list with appropriate Wikipedia URLs
# for your topic(s) of interest

# ensure that NLTK has been installed along with the stopwords corpora
# pip install nltk
# python -m nltk.downloader stopwords

import scrapy
import os.path
from WebFocusedCrawl.items import WebfocusedcrawlItem  # item class 
import nltk  # used to remove stopwords from tags
import re  # regular expressions used to parse tags

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    good_tokens = [token for token in tokens if token not in stopword_list]
    return good_tokens     

class ArticlesSpider(scrapy.Spider):
    name = "articles-spider"

    def start_requests(self):
        allowed_domains = ['en.wikipedia.org']

        # list of Wikipedia URLs for topic of interest
        urls = [
            "https://en.wikipedia.org/wiki/Climate_change",
            "https://en.wikipedia.org/wiki/Scientific_consensus_on_climate_change",
            "https://en.wikipedia.org/wiki/History_of_climate_change_science",
            "https://en.wikipedia.org/wiki/Effects_of_climate_change",
            "https://en.wikipedia.org/wiki/Greenhouse_gas",
            "https://en.wikipedia.org/wiki/Global_warming_(disambiguation)",
            "https://en.wikipedia.org/wiki/Greenhouse_gas_emissions",
            "https://simple.wikipedia.org/wiki/Global_warming",
            "https://en.wikipedia.org/wiki/Global_warming_controversy",
            "https://en.wikipedia.org/wiki/2021_in_climate_change",
            "https://climate.nasa.gov",
            "https://www.ipcc.ch/",
            "https://www.epa.gov/climate-change",
            "https://www.nrdc.org/stories/global-climate-change-what-you-need-know",
            "https://en.wikipedia.org/wiki/Climate_variability_and_change",
            "https://en.wikipedia.org/wiki/Long-term_effects_of_climate_change",
            "https://en.wikipedia.org/wiki/Attribution_of_recent_climate_change",
            "https://www.washingtonpost.com/weather/2021/10/06/fall-foliage-leaves-climate-change/"
            ]
            
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # first part: save wikipedia page html to wikipages directory
        page = response.url.split("/")[4]
        page_dirname = 'wikipages'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename) 

        # follow links
        linksraw = response.css('a::attr(href)').getall()
        links = ','.join(linksraw)

        for link in response.css('a::attr(href)'):
            url = response.urljoin(link.extract())
            title = response.css('title::text').get()
            yield scrapy.Request(url=url, callback=self.parse_link)
        
        # second part: extract text for the item for document corpus
        item = WebfocusedcrawlItem()
        item['url'] = response.url
        item['title'] = response.css('h1::text').extract_first()
        item['text'] = response.xpath('//div[@id="mw-content-text"]//text()')\
                           .extract()                                                             
        tags_list = [response.url.split("/")[2],
                     response.url.split("/")[3]]
        more_tags = [x.lower() for x in remove_stopwords(response.url\
                       	    .split("/")[4].split("_"))]
        for tag in more_tags:
            tag = re.sub('[^a-zA-Z]', '', tag)  # alphanumeric values only  
            tags_list.append(tag)
        item['tags'] = tags_list
        return item 

    def parse_link(self, response):
        # first part: save wikipedia page html to wikipages directory
        page = response.url.split("/")[4]
        page_dirname = 'wikipages'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename) 
        
        # second part: extract text for the item for document corpus
        item = WebfocusedcrawlItem()
        item['url'] = response.url
        item['title'] = response.css('h1::text').extract_first()
        item['text'] = response.xpath('//div[@id="mw-content-text"]//text()')\
                           .extract()                                                             
        tags_list = [response.url.split("/")[2],
                     response.url.split("/")[3]]
        more_tags = [x.lower() for x in remove_stopwords(response.url\
                       	    .split("/")[4].split("_"))]
        for tag in more_tags:
            tag = re.sub('[^a-zA-Z]', '', tag)  # alphanumeric values only  
            tags_list.append(tag)
        item['tags'] = tags_list
        return item 
