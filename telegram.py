import logging
import os
import re
from parser import parser
from model_base.model_base import get_df,word_cloud_output,tf_idf,similar_comments
import spacy
import requests
from aiogram import Bot, Dispatcher, executor, types
import math
from static_text import (HELLO_TEXT, NOT_TARGET_CONTENT_TYPES, NOT_TARGET_TEXT, NOT_RESPONSE_LINK,
                         NOT_TARGET_TEXT_LINK, WAITING_TEXT, FINAL_TEXT, HELP_TEXT, SMALL_COMMENTS)

# Comfigure logging
logging.basicConfig(level = logging.INFO)

# Make sure that you got telegram api token from BotFather
# перенести в энв!!!

TOKEN = '1740386855:AAHgxBqSAaCalAOh0j1K0Afhd1hsi-4qrqc'
#TOKEN = os.getenv('TOKEN_API_DS_BOT')


# Initialize bot and dispetcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
nlp = spacy.load('ru_core_news_lg')


#Base command messages for start and exceptions(not target content inputs)
@dp.message_handler(commands = ['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT %user_name
    logging.info(f'first start from user_name = {user_name}, user_id = {user_id}')
    
    await message.reply(text)


@dp.message_handler(commands = ['help'])
async def send_help(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELP_TEXT %user_name
    logging.info(f'help from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)    

@dp.message_handler(content_types = NOT_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = NOT_TARGET_TEXT %user_name
    logging.info(f'Not_target_content_type from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)


# @dp.message_handler(commands = ['sendlink'])
@dp.message_handler(content_types=['text'])
async def handle_link(message):
  #  url = 'http://api:8000/api/photo'
    chat_id = message.chat.id
    text_link = message.text
    user_name = message.from_user.first_name
    user_id = message.from_user.id

    res_id = 0
    for i in text_link.split('/'):
        try:
            if float(i) and math.floor(float(i)) == int(i):
                res_id = int(i)
                break 
        except:
            pass   
    logging.info(f'Receive text from user_name = {user_name}, user_id = {user_id},item_id = {res_id}')
    
    if not os.path.isfile(f'./df/{user_id}.csv') or res_id != 0:              
        text_template = r'wildberries\.ru\/catalog\/\d*'
        result = re.findall(text_template, text_link)
        if len(result) == 1:        
            final_link = f'https://www.{result[0]}/otzyvy'

            if requests.get(''.join(final_link)).status_code != 200:
                user_name = message.from_user.first_name
                user_id = message.from_user.id
                text = NOT_RESPONSE_LINK %user_name
                logging.info(f'{user_name, user_id} send not found link')
                await message.reply(text)
                
            else:
                user_name = message.from_user.first_name
                user_id = message.from_user.id
                message_id = message.message_id
                text = WAITING_TEXT %user_name
                logging.info(f'{user_name, user_id} is knocking to our bot')
                await bot.send_message(chat_id, text)
                #sku из html кода
                sku_html = final_link[35:-7]
                file_name = f'./input/file_{sku_html}_{user_id}_{message_id}.jl'
                
                parser(final_link, file_name)
                text = FINAL_TEXT
                await bot.send_message(chat_id, text)
            
                get_df(file_name,user_id)
                if not os.path.isfile(f'./df/{user_id}.csv'):
                    text = SMALL_COMMENTS  %user_name
                    logging.info(f'{user_name, user_id} semd link with small count of comments')
                    await message.reply(text)
                else:

                    output_name=f'./output/plot_{file_name[8:-2]}.jpg'
                    preprocessed_comments = word_cloud_output(file_name,output_name,user_id)           
                    await bot.send_photo(chat_id, photo=open(output_name,'rb'))

                    text = tf_idf(preprocessed_comments)
                    await bot.send_message(chat_id, text)
                
        else:
            text = NOT_TARGET_TEXT_LINK %user_name
            await message.reply(text)  
    else:
        chat_id = message.chat.id        
        user_name = message.from_user.first_name
        user_id = message.from_user.id      
        message_id = message.message_id          
        word = message.text.lower()
        logging.info(f'{user_name, user_id} send word:{word}')                            
        text = similar_comments(word,nlp,user_id)
        logging.info(f'{user_name, user_id} receive text:{text}')
        await bot.send_message(chat_id, text)            
                
   
            #  модель


        # Define input photo local path
      #  photo_name = './input/photo_%s_%s.jpg' %(user_id, message_id)
    #    photo = await message.photo[-1].download(photo_name)  # extract photo for further procceses
     #   with open(photo_name, 'rb') as f:
    #        res = requests.post(url, files={'photo': f})
     #       dog_prob = res.json()

     #   await bot.send_message(chat_id,dog_prob)

  
                
     
       
          
           



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
