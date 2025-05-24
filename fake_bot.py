import io
import asyncio
import json
import os
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from datetime import datetime
from redis import asyncio as aioredis
from PIL import Image, ImageDraw, ImageFont
import random
import time

# Конфигурация
API_TOKEN = '7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ'
ADMIN_CHAT_ID = 234037002
REDIS_URL = "redis://localhost:6379/0"
FONT_PATH = "arial.ttf"  # Укажите путь к шрифту
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "example")  # Путь к папке example

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class Form(StatesGroup):
    waiting_for_style = State()
    waiting_for_text = State()

def get_local_images(text, style):
    """Получение изображений из локальной папки с сортировкой по названию"""
    images = []
    stories = []
    
    # Проверяем существование папки
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        return [], [], [], ["Папка с изображениями пуста"]
    
    # Получаем список файлов в папке и сортируем по названию
    image_files = sorted(
        [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: x.lower()
    )
    
    # Если файлов нет, возвращаем пустой список
    if not image_files:
        return [], [], [], ["Папка с изображениями пуста"]
    
    # Выбираем первые 3 изображения (после сортировки)
    selected_files = image_files[:3]
    
    # Стильные нуар-подписи для изображений
    noir_captions = [
        "Чёрная машина под дождём. В салоне — кожаный портфель с деньгами и пистолетом.",
        "Двое выходят из тени. Без слов обмениваются взглядами. В воздухе пахнет изменой и дождём.",
        "Машина исчезает в ночи, оставляя только следы на мокром асфальте и неотвеченные вопросы."
    ]
    
    for i, filename in enumerate(selected_files):
        try:
            img_path = os.path.join(IMAGE_FOLDER, filename)
            img = Image.open(img_path)
            
            # Добавляем текст на изображение
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(FONT_PATH, 40)
            except:
                font = ImageFont.load_default()
            
            # Позиция текста зависит от размера изображения
            width, height = img.size
            text_position = (width//10, height//10)
            
            # Добавляем нуар-подпись в угол изображения
            draw.text(text_position, noir_captions[i], 
                     font=font, fill=(255, 255, 255))
            
            images.append(img)
            stories.append(noir_captions[i] + f"\n\nФайл: {filename}")
        except Exception as e:
            print(f"Ошибка обработки изображения {filename}: {e}")
    
    # Генерируем фейковые метрики
    clips = [random.uniform(0.7, 0.9) for _ in range(len(images))]
    lpips = [random.uniform(0.2, 0.4) for _ in range(len(images))]
    
    return images, clips, lpips, stories

async def on_startup(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🤖 Бот для отправки локальных изображений запущен!")
    asyncio.create_task(task_consumer())

async def on_shutdown(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🔴 Бот остановлен")
    redis = await aioredis.from_url(REDIS_URL)
    await redis.close()

async def task_consumer():
    redis = await aioredis.from_url(REDIS_URL)
    while True:
        _, task_data = await redis.blpop('image_queue')
        task = json.loads(task_data)
        await process_image_task(task)

async def process_image_task(task):
    try:
        user_id = task['user_id']
        chat_id = task['chat_id']
        text = task['text']
        style = task['style']
        
        # Добавляем случайную задержку от 15 до 30 секунд
        delay = random.randint(15, 30)
        await bot.send_message(chat_id, f"⏳ Обработка вашего запроса займет {delay} секунд...")
        await asyncio.sleep(delay)
        
        await bot.send_message(chat_id, "🚀 Подготовка ваших изображений...")
        
        # Получаем изображения из локальной папки
        result = get_local_images(text, style)
        best_frames, best_clips, best_lpips, story_ls = result
        
        if not best_frames:
            await bot.send_message(chat_id, "❌ В папке нет изображений для отправки")
            return
        
        # Отправка результатов
        for i in range(len(story_ls)):
            caption = (f"🎨 Стиль: {'Midjourney' if style else 'Обычный'}\n"
                      f"📖 {story_ls[i]}\n\n"
                      f"📊 Метрики:\n🖼 CLIP: {best_clips[i]:.2f}\n📐 LPIPS: {best_lpips[i]:.2f}")
            
            img_byte_arr = io.BytesIO()
            best_frames[i].save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            await bot.send_photo(chat_id, types.InputFile(img_byte_arr), caption=caption)
            await asyncio.sleep(1)  # Небольшая пауза между отправкой изображений
        
        await bot.send_message(chat_id, "✅ Изображения отправлены! /start для нового запроса")
    
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Ошибка: {str(e)}")
        await bot.send_message(ADMIN_CHAT_ID, f"Ошибка у {user_id}: {str(e)}")

@dp.message_handler(commands=['start', 'help'])
async def cmd_start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add("Да", "Нет")
    await message.answer("Использовать стиль Midjourney?", reply_markup=keyboard)
    await Form.waiting_for_style.set()

@dp.message_handler(state=Form.waiting_for_style)
async def process_style(message: types.Message, state: FSMContext):
    if message.text.lower() not in ['да', 'нет']:
        return await message.answer("Ответьте Да/Нет")
    
    await state.update_data(style=message.text.lower() == 'да')
    await message.answer("Введите текст (≥50 символов):", reply_markup=types.ReplyKeyboardRemove())
    await Form.next()

@dp.message_handler(state=Form.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    if len(message.text) < 50:
        return await message.answer("❌ Слишком короткий текст")
    
    data = await state.get_data()
    await state.finish()
    
    redis = await aioredis.from_url(REDIS_URL)
    task = {
        'user_id': message.from_user.id,
        'chat_id': message.chat.id,
        'text': message.text,
        'style': data['style']
    }
    
    queue_position = await redis.rpush('image_queue', json.dumps(task))
    await message.answer(
        f"⏳ Запрос принят в обработку.\n"
        f"📍 Ваша позиция в очереди: {queue_position}\n"
        f"⏱ Ожидайте начала обработки..."
    )

if __name__ == '__main__':
    # Проверяем существование папки example при запуске
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"Создана папка для изображений: {IMAGE_FOLDER}")
    
    executor.start_polling(dp, 
                         on_startup=on_startup,
                         on_shutdown=on_shutdown,
                         skip_updates=True)
