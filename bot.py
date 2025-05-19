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
import aioredis
from story_image import *

# Конфигурация
API_TOKEN = 'YOUR_BOT_TOKEN'
ADMIN_CHAT_ID = 234037002
REDIS_URL = "redis://localhost"
MAX_WORKERS = 4  # Максимальное количество параллельных задач

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

class Form(StatesGroup):
    waiting_for_style = State()
    waiting_for_text = State()

async def on_startup(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🤖 Бот запущен!")
    asyncio.create_task(task_consumer())

async def on_shutdown(dp):
    await bot.send_message(ADMIN_CHAT_ID, "🔴 Бот остановлен")
    redis = await aioredis.from_url(REDIS_URL)
    await redis.close()

async def task_consumer():
    """Фоновый процесс для обработки задач из очереди"""
    redis = await aioredis.from_url(REDIS_URL)
    while True:
        _, task_data = await redis.blpop('image_queue')
        task = json.loads(task_data)
        await process_image_task(task)

async def process_image_task(task):
    """Обработка одной задачи генерации изображений"""
    try:
        user_id = task['user_id']
        chat_id = task['chat_id']
        text = task['text']
        style = task['style']
        
        # Запуск блокирующей операции в отдельном потоке
        result = await asyncio.get_event_loop().run_in_executor(
            executor_pool,
            lambda: gen_img(text, style)
        )
        
        best_frames, best_clips, best_lpips, story_ls = result
        
        # Отправка результатов
        for i in range(len(story_ls)):
            caption = (
                f"🎨 Стиль: {'Midjourney' if style else 'Обычный'}\n"
                f"📖 {story_ls[i]}\n\n"
                f"📊 Метрики:\n🖼 CLIP: {best_clips[i]:.2f}\n📐 LPIPS: {best_lpips[i]:.2f}"
            )
            with io.BytesIO() as output:
                best_frames[i].save(output, format='JPEG')
                output.seek(0)
                await bot.send_photo(chat_id, types.InputFile(output), caption=caption)
                await asyncio.sleep(0.1)
        
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
    
    # Добавляем задачу и получаем текущую длину очереди
    queue_position = await redis.rpush('image_queue', json.dumps(task))
    
    # Отправляем пользователю его позицию в очереди
    await message.answer(
        f"⏳ Запрос принят в обработку.\n"
        f"📍 Ваша позиция в очереди: {queue_position}\n"
        f"⏱ Ожидайте начала генерации..."
    )

async def process_image_task(task):
    """Обработка одной задачи генерации изображений"""
    try:
        user_id = task['user_id']
        chat_id = task['chat_id']
        text = task['text']
        style = task['style']
        
        # Уведомляем о начале обработки
        await bot.send_message(chat_id, "🚀 Начинаю генерацию вашего изображения...")


if __name__ == '__main__':
    executor.start_polling(dp, 
                         on_startup=on_startup,
                         on_shutdown=on_shutdown,
                         skip_updates=True)
