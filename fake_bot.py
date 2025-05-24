import io
import asyncio
import json
import concurrent.futures
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from datetime import datetime
from redis import asyncio as aioredis
from PIL import Image, ImageDraw, ImageFont
import random

# Конфигурация
API_TOKEN = '7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ'
ADMIN_CHAT_ID = 234037002
REDIS_URL = "redis://localhost"
MAX_WORKERS = 4
FONT_PATH = "arial.ttf"  # Укажите путь к шрифту

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

class Form(StatesGroup):
    waiting_for_style = State()
    waiting_for_text = State()

def fake_gen_img(text, style):
    """Генерация фейковых изображений с текстом"""
    images = []
    stories = []
    
    # Создаем 3 тестовых варианта
    for i in range(3):
        # Создаем изображение 512x512
        img = Image.new('RGB', (512, 512), color=(random.randint(0, 255), 
                       random.randint(0, 255), random.randint(0, 255)))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(FONT_PATH, 40)
        except:
            font = ImageFont.load_default()
            
        text_position = (50, 256-20)
        d.text(text_position, f"Вариант {i+1}\n{text[:20]}...", 
              font=font, fill=(255, 255, 255))
        
        images.append(img)
        stories.append(f"Фейковая история {i+1} для текста: {text[:50]}...")
    
    # Генерируем фейковые метрики
    clips = [random.uniform(0.7, 0.9) for _ in range(3)]
    lpips = [random.uniform(0.2, 0.4) for _ in range(3)]
    
    return images, clips, lpips, stories

async def on_startup(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🤖 Фейк-бот запущен!")
    asyncio.create_task(task_consumer())

async def on_shutdown(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🔴 Фейк-бот остановлен")
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
        
        await bot.send_message(chat_id, "🚀 Начинаю генерацию вашего изображения...")
        
        # Запускаем фейковую генерацию
        result = await asyncio.get_event_loop().run_in_executor(
            executor_pool,
            lambda: fake_gen_img(text, style)
        )
        
        best_frames, best_clips, best_lpips, story_ls = result
        
        # Имитация обработки
        await asyncio.sleep(2)
        
        # Отправка результатов
        for i in range(len(story_ls)):
            caption = (f"🎨 Стиль: {'Midjourney' if style else 'Обычный'}\n"
                      f"📖 {story_ls[i]}\n\n"
                      f"📊 Метрики:\n🖼 CLIP: {best_clips[i]:.2f}\n📐 LPIPS: {best_lpips[i]:.2f}")
            
            img_byte_arr = io.BytesIO()
            best_frames[i].save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            await bot.send_photo(chat_id, types.InputFile(img_byte_arr), caption=caption)
            await asyncio.sleep(0.5)
        
        await bot.send_message(chat_id, "✅ Генерация завершена! /start для нового запроса")
    
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
        f"⏱ Ожидайте начала генерации..."
    )

if __name__ == '__main__':
    executor.start_polling(dp, 
                         on_startup=on_startup,
                         on_shutdown=on_shutdown,
                         skip_updates=True)
