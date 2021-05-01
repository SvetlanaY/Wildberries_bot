import scrapy
from ..items import WildberriesItem
from scrapy.http import JsonRequest
import json




class WildberriesSpider(scrapy.Spider):
    name = 'wildberries'
    allowed_domains = ['wildberries.ru']

    #проверить входящую ссылку! меняется номер
   # start_urls = ['https://www.wildberries.ru/catalog/10777032/otzyvy'] 
   # start_urls = final_linc 
    
    count_comment = 10000
    
      
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

        

        
        # Если сайт отдает html то так:
        # xpath можно узнать в панели отладки хрома (правой кнопкой мышки на элементе), например:
     #   xpath_name = '//*[@id="global"]/div/table/tbody/tr/td[%(col)s]/table/tbody/tr/td/a/text()'
     #   hxs = HtmlXPathSelector(response)
     #   column = 100500
      #  data_from_html = hxs.select(xpath_name % {'col': column}).extract()
       # yield data_from_json['feedbacks'][0]
      # yield result
           

