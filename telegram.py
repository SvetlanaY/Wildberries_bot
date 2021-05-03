import logging
import os
import re
from parser import parser

import requests
from aiogram import Bot, Dispatcher, executor, types

from static_text import (HELLO_TEXT, NOT_TARGET_CONTENT_TYPES, NOT_TARGET_TEXT,
                         NOT_TARGET_TEXT_LINK, WAITING_TEXT, FINAL_TEXT)

# Comfigure logging
logging.basicConfig(level = logging.INFO)

# Make sure that you got telegram api token from BotFather
# перенести в энв!!!

TOKEN = '1740386855:AAHgxBqSAaCalAOh0j1K0Afhd1hsi-4qrqc'
#TOKEN = os.getenv('TOKEN_API_DS_BOT')

# Initialize bot and dispetcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

#Base command messages for start and exceptions(not target content inputs)
@dp.message_handler(commands = ['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT %user_name
    logging.info(f'first start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)

@dp.message_handler(content_types = NOT_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    text = NOT_TARGET_TEXT %user_name
    await message.reply(text)

@dp.message_handler(content_types=['text'])
async def handle_photo_for_prediction(message):
  #  url = 'http://api:8000/api/photo'
    chat_id = message.chat.id
    text_link = message.text
    text_template = r'wildberries\.ru\/catalog\/\d*'
    result = re.findall(text_template, text_link)

    if len(result) == 1:
        final_link = []
        final_link.append(f'https://www.{result[0]}/otzyvy')

        if requests.get(''.join(final_link)).status_code != 200:
            user_name = message.from_user.first_name
            text = NOT_TARGET_TEXT_LINK %user_name
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
            text = FINAL_TEXT %user_name
            await bot.send_message(chat_id, text)
            #  модель


        # Define input photo local path
      #  photo_name = './input/photo_%s_%s.jpg' %(user_id, message_id)
    #    photo = await message.photo[-1].download(photo_name)  # extract photo for further procceses
     #   with open(photo_name, 'rb') as f:
    #        res = requests.post(url, files={'photo': f})
     #       dog_prob = res.json()

     #   await bot.send_message(chat_id,dog_prob)

    else:
        user_name = message.from_user.first_name
        text = NOT_TARGET_TEXT_LINK %user_name
        await message.reply(text)



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
