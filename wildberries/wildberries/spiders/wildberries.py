import json

import scrapy
from scrapy.http import JsonRequest

from ..items import WildberriesItem


class WildberriesSpider(scrapy.Spider):
    name = 'wildberries'
    allowed_domains = ['wildberries.ru']
   
    count_comment = 5000    
      
    def parse(self, response):       
        id_api = int(response.css('div::attr(data-good-link)').get())
        data = {"imtId":id_api,"take":self.count_comment,"order":"dateDesc"}
        yield JsonRequest(url='https://public-feedbacks.wildberries.ru/api/v1/feedbacks/site', data=data,callback=self.parse_my_url)
     
 
    def parse_my_url(self, response):
        # Если сайт отдает ответ в виде json:
        data_from_json = json.loads(response.body)
        result={}       
        
        for i in range(self.count_comment):
            try :
                res=[]
                res.append(data_from_json['feedbacks'][i].get('text'))
                res.append(data_from_json['feedbacks'][i].get('createdDate'))
                res.append(data_from_json['feedbacks'][i].get('color'))
                res.append(data_from_json['feedbacks'][i].get('size'))
                res.append(data_from_json['feedbacks'][i].get('votes').get('pluses'))
                res.append(data_from_json['feedbacks'][i].get('votes').get('minuses'))
                res.append(data_from_json['feedbacks'][i].get('productValuation'))
                res.append(data_from_json['feedbacks'][i].get('productDetails').get('productName'))
                res.append(data_from_json['feedbacks'][i].get('productDetails').get('brandName'))
                result[i]=res
            except:
                pass
            
        yield result
        
