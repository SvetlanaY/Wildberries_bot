import logging
import math
import os
import re
from parser import parser

import requests
import spacy
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove)

from model_base.model_base import (get_df, similar_comments, similar_comments_,
                                   tf_idf, word_cloud_output)
from static_text import (FINAL_TEXT, HELLO_TEXT, HELP_TEXT, NOT_RESPONSE_LINK,
                         NOT_TARGET_CONTENT_TYPES, NOT_TARGET_TEXT, CHANGE_QUERY_TEXT,
                         NOT_TARGET_TEXT_LINK, SMALL_COMMENTS, WAITING_TEXT)

# Comfigure logging
# logging.basicConfig(level = logging.INFO)
logging.basicConfig(filename='app.log', level=logging.INFO, filemode='w')


# Make sure that you got telegram api token from BotFather
# перенести в энв!!!

TOKEN = '1740386855:AAHgxBqSAaCalAOh0j1K0Afhd1hsi-4qrqc'
# TOKEN = os.getenv('TOKEN_API_DS_BOT')


# Initialize bot and dispetcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
nlp = spacy.load('ru_core_news_lg')


# Base command messages for start and exceptions(not target content inputs)
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT % user_name
    logging.info(
        f'first start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELP_TEXT % user_name
    logging.info(f'help from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)


@dp.message_handler(content_types=NOT_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = NOT_TARGET_TEXT % user_name
    logging.info(
        f'Not_target_content_type from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)


@dp.message_handler(content_types=['text'])
async def handle_link(message):
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
        except BaseException:
            pass
    logging.info(
        f'Receive text from user_name = {user_name}, user_id = {user_id},item_id = {res_id}')

    if not os.path.isfile(f'./df/{user_id}.csv') or res_id != 0:
        text_template = r'wildberries\.ru\/catalog\/\d*'
        result = re.findall(text_template, text_link)
        if len(result) == 1:
            final_link = f'https://www.{result[0]}/otzyvy'
            if requests.get(''.join(final_link)).status_code != 200:
                user_name = message.from_user.first_name
                user_id = message.from_user.id
                text = NOT_RESPONSE_LINK % user_name
                logging.info(f'{user_name, user_id} send not found link')
                await message.reply(text)
            else:
                user_name = message.from_user.first_name
                user_id = message.from_user.id
                message_id = message.message_id
                text = WAITING_TEXT % user_name
                logging.info(f'{user_name, user_id} is knocking to our bot')
                await bot.send_message(chat_id, text)
                # sku из html кода
                sku_html = final_link[35:-7]
                file_name = f'./input/file_{sku_html}_{user_id}_{message_id}.jl'
                parser(final_link, file_name)
                text = FINAL_TEXT
                await bot.send_message(chat_id, text)

                get_df(file_name, user_id)
                if not os.path.isfile(f'./df/{user_id}.csv'):
                    text = SMALL_COMMENTS % user_name
                    logging.info(
                        f'{user_name, user_id} semd link with small count of comments')
                    await message.reply(text, reply_markup=ReplyKeyboardRemove())

                else:
                    text, tf_idf_indexes = tf_idf(file_name, user_id)
                    output_name = f'./output/plot_{file_name[8:-2]}.jpg'
                    word_cloud_output(tf_idf_indexes, output_name)
                    await bot.send_photo(chat_id, photo=open(output_name, 'rb'))
                    if os.path.isfile(output_name):
                        os.remove(output_name)
                    if text[0] != 'Мало комментариев':
                        buttons_k = []
                        for i in range(len(text)):
                            buttons_k.append(KeyboardButton(text[i]))
                        markup = ReplyKeyboardMarkup(
                            row_width=1, resize_keyboard=True)
                        markup.add(*buttons_k)
                        await message.answer('Нажми на слово или напиши что-то своё',
                                             reply_markup=markup)
                    else:
                        await message.reply(text, reply_markup=ReplyKeyboardRemove())

        else:
            text = NOT_TARGET_TEXT_LINK % user_name
            await message.reply(text)
    else:
        chat_id = message.chat.id
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        message_id = message.message_id
        word = message.text.lower()

        # сохраняем датафрейм комментов по топикам
        similar_comments_(word, nlp, user_id)
        if not os.path.isfile(f'./similarity/{user_id}_{word}.csv'):
            await message.reply(CHANGE_QUERY_TEXT)

        word_pos = word + 'p'
        word_neg = word + 'n'
        word_neu = word + 'a'

        #  кнопки sentiments
        button_sent_p = InlineKeyboardButton(
            'Положительные отзывы',
            callback_data=f'sent_{word_pos}')
        button_sent_n = InlineKeyboardButton(
            'Отрицательные отзывы',
            callback_data=f'sent_{word_neg}')
        button_sent_all = InlineKeyboardButton(
            'Все отзывы', callback_data=f'sent_{word_neu}')
        markup_sent = InlineKeyboardMarkup(row_width=1)
        markup_sent.add(button_sent_p, button_sent_n, button_sent_all)
        await message.answer('Какие отзывы тебе нужны?', reply_markup=markup_sent)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('sent_'))
async def callback_sent(call: types.CallbackQuery):
    action = call.data.split('_')[1]
    user_name = call.from_user.first_name
    user_id = call.from_user.id
    logging.info(f'{user_name, user_id} send word:{action}')
    text = similar_comments(action, nlp, user_id)
    logging.info(f'{user_name, user_id} receive text:{text}')
    await call.message.answer(text)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('sim_'))
async def callback_top(call: types.CallbackQuery):
    action = call.data.split('_')[1]
    user_name = call.from_user.first_name
    user_id = call.from_user.id
    logging.info(f'{user_name, user_id} send word:{action}')
    text = similar_comments(action, nlp, user_id)
    logging.info(f'{user_name, user_id} receive text:{text}')
    await call.message.answer(text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
