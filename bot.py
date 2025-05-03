import io
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from story_image import *

# Замените на ваш токен
API_TOKEN = '7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ'

# # Инициализация бота
# bot = Bot(token=API_TOKEN)
# storage = MemoryStorage()
# dp = Dispatcher(bot, storage=storage)

# class Form(StatesGroup):
#     waiting_for_text = State()

# @dp.message_handler(commands=['start'])
# async def cmd_start(message: types.Message):
#     await message.answer("Привет! Отправь мне историю, и я сгенерирую для тебя иллюстрации к ней.")
#     await Form.waiting_for_text.set()

# @dp.message_handler(state=Form.waiting_for_text)
# async def process_text(message: types.Message, state: FSMContext):
#     user_text = message.text
    
#     # Получаем данные от функции gen_img
#     best_frames, best_clips, best_lpips, story_ls = gen_img(user_text)
    
#     # Проверяем, что все списки одинаковой длины
#     if not (len(best_frames) == len(best_clips) == len(best_lpips) == len(story_ls)):
#         await message.answer("Ошибка: получены списки разной длины")
#         await state.finish()
#         return
    
#     # Отправляем пользователю все сгенерированные сообщения
#     for i in range(len(story_ls)):
#         # Формируем текст сообщения с метриками
#         msg_text = f"{story_ls[i]}\n\nМетрики:\nCLIP: {best_clips[i]}\nLPIPS: {best_lpips[i]}"
        
#         # Конвертируем изображение в байты
#         img_byte_arr = io.BytesIO()
#         best_frames[i].save(img_byte_arr, format='JPEG')
#         img_byte_arr.seek(0)
        
#         # Отправляем фото с подписью
#         await message.answer_photo(types.InputFile(img_byte_arr), caption=msg_text)
    
#     await state.finish()

# if __name__ == '__main__':
#     executor.start_polling(dp, skip_updates=True)

import io
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from datetime import datetime

# # Замените на ваш токен
# API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

# Инициализация бота
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Статус работы бота
bot_start_time = datetime.now()

async def on_startup(dp):
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text="🤖 Бот успешно запущен!")
    print(f"Бот запущен в {bot_start_time}")

async def on_shutdown(dp):
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text="🔴 Бот остановлен")
    print("Бот остановлен")

# Предполагаемая функция gen_img (замените на вашу реальную функцию)
def gen_img(text: str):
    # Здесь ваша логика обработки текста и генерации данных
    # Пример возвращаемых данных (замените на реальные)
    from PIL import Image, ImageDraw
    
    # Создаем тестовые изображения
    images = []
    for i in range(4):
        img = Image.new('RGB', (256, 256), color=(i*50, i*30, i*70))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Image {i+1}", fill=(255, 255, 0))
        images.append(img)
    
    best_frames = images
    best_clips = [0.9, 0.8, 0.7, 0.6]
    best_lpips = [0.1, 0.2, 0.3, 0.4]
    story_ls = [
        "История 1 по вашему тексту",
        "История 2 по вашему тексту",
        "История 3 по вашему тексту",
        "История 4 по вашему тексту"
    ]
    return best_frames, best_clips, best_lpips, story_ls

class Form(StatesGroup):
    waiting_for_text = State()

@dp.message_handler(commands=['start', 'help'])
async def cmd_start(message: types.Message):
    await message.answer("🤖 Привет! Я бот для генерации изображений по тексту.\n"
                        "Просто отправь мне любой текст, и я создам для тебя уникальные изображения с историями!\n\n"
                        f"🟢 Бот работает с {bot_start_time.strftime('%d.%m.%Y %H:%M')}")
    await Form.waiting_for_text.set()

@dp.message_handler(commands=['status'])
async def cmd_status(message: types.Message):
    uptime = datetime.now() - bot_start_time
    await message.answer(f"🟢 Бот в работе\n"
                        f"⏱ Время работы: {str(uptime).split('.')[0]}\n"
                        f"📅 Запущен: {bot_start_time.strftime('%d.%m.%Y %H:%M:%S')}")

@dp.message_handler(state=Form.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    user_text = message.text
    
    # Отправляем статус "печатает..."
    status_msg = await message.answer("🖨 Обрабатываю ваш запрос...")
    
    try:
        # Имитируем обработку (в реальном коде замените на gen_img)
        await asyncio.sleep(1)  # Заглушка для имитации работы
        
        # Получаем данные от функции gen_img
        best_frames, best_clips, best_lpips, story_ls = gen_img(user_text)
        
        # Проверяем, что все списки одинаковой длины
        if not (len(best_frames) == len(best_clips) == len(best_lpips) == len(story_ls)):
            await message.answer("❌ Ошибка: получены списки разной длины")
            await state.finish()
            return
        
        # Удаляем статус "печатает"
        await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
        
        # Отправляем статус генерации
        progress_msg = await message.answer(f"🔍 Сгенерировано {len(story_ls)} вариантов...")
        
        # Отправляем пользователю все сгенерированные сообщения
        for i in range(len(story_ls)):
            # Формируем текст сообщения с метриками
            msg_text = f"📖 {story_ls[i]}\n\n📊 Метрики:\n🖼 CLIP: {best_clips[i]:.2f}\n📐 LPIPS: {best_lpips[i]:.2f}"
            
            # Конвертируем изображение в байты
            img_byte_arr = io.BytesIO()
            best_frames[i].save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Отправляем фото с подписью
            await message.answer_photo(types.InputFile(img_byte_arr), caption=msg_text)
            
            # Небольшая задержка между сообщениями
            await asyncio.sleep(0.5)
        
        # Удаляем сообщение о прогрессе
        await bot.delete_message(chat_id=message.chat.id, message_id=progress_msg.message_id)
        
        # Финальное сообщение
        await message.answer("✅ Генерация завершена!\n"
                           "Хотите попробовать ещё раз? Просто отправьте новый текст.")
    
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")
        #await bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Ошибка в боте: {str(e)}")
    
    finally:
        await state.finish()

if __name__ == '__main__':
    # Замените на ваш chat_id для уведомлений
    #ADMIN_CHAT_ID = 123456789
    
    executor.start_polling(dp, 
                         on_startup=on_startup, 
                         on_shutdown=on_shutdown, 
                         skip_updates=True)
