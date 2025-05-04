import io
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from datetime import datetime
from story_image import *
API_TOKEN = '7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ'
ADMIN_CHAT_ID = 234037002

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
bot_start_time = datetime.now()

class Form(StatesGroup):
    waiting_for_style = State()
    waiting_for_text = State()

async def on_startup(dp):
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text="🤖 Бот успешно запущен!")

async def on_shutdown(dp):
    await bot.send_message(chat_id=ADMIN_CHAT_ID, text="🔴 Бот остановлен")

@dp.message_handler(commands=['start', 'help'])
async def cmd_start(message: types.Message):
    await message.answer("🤖 Привет! Я бот для генерации изображений по тексту.\n\n"
                        "Сначала нужно выбрать стиль генерации:")
    
    # Создаем клавиатуру с кнопками
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("Да"), types.KeyboardButton("Нет"))
    
    await message.answer("Использовать стиль Midjourney? (Да/Нет)", reply_markup=keyboard)
    await Form.waiting_for_style.set()

@dp.message_handler(state=Form.waiting_for_style)
async def process_style(message: types.Message, state: FSMContext):
    style_choice = message.text.lower()
    
    if style_choice not in ['да', 'нет']:
        await message.answer("Пожалуйста, ответьте 'Да' или 'Нет'")
        return
    
    midjourney_style = style_choice == 'да'
    
    # Сохраняем выбор стиля
    await state.update_data(midjourney_style=midjourney_style)
    
    # Убираем клавиатуру
    await message.answer("Теперь введите текст для генерации изображений (не менее 50 символов):", 
                        reply_markup=types.ReplyKeyboardRemove())
    
    await Form.waiting_for_text.set()

@dp.message_handler(state=Form.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    user_text = message.text
    
    # Проверка длины текста
    if len(user_text) < 50:
        await message.answer("❌ Текст слишком короткий. Пожалуйста, введите не менее 50 символов.")
        return
    
    # Получаем сохраненный выбор стиля
    user_data = await state.get_data()
    midjourney_style = user_data.get('midjourney_style', False)
    
    status_msg = await message.answer("🖨 Обрабатываю ваш запрос...")
    
    try:
        # Получаем данные от функции gen_img
        best_frames, best_clips, best_lpips, story_ls = gen_img(user_text, midjourney_style)
        
        if not (len(best_frames) == len(best_clips) == len(best_lpips) == len(story_ls)):
            await message.answer("❌ Ошибка: получены списки разной длины")
            return
        
        await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
        
        progress_msg = await message.answer(f"🔍 Сгенерировано {len(story_ls)} вариантов...")
        
        for i in range(len(story_ls)):
            msg_text = (f"🎨 Стиль: {'Midjourney' if midjourney_style else 'Обычный'}\n"
                       f"📖 {story_ls[i]}\n\n"
                       f"📊 Метрики:\n🖼 CLIP: {best_clips[i]:.2f}\n📐 LPIPS: {best_lpips[i]:.2f}")
            
            img_byte_arr = io.BytesIO()
            best_frames[i].save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            await message.answer_photo(types.InputFile(img_byte_arr), caption=msg_text)
            await asyncio.sleep(0.5)
        
        await bot.delete_message(chat_id=message.chat.id, message_id=progress_msg.message_id)
        await message.answer("✅ Генерация завершена! Хотите создать ещё? Нажмите /start")
    
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Ошибка в боте: {str(e)}")
    
    finally:
        await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, 
                         on_startup=on_startup, 
                         on_shutdown=on_shutdown, 
                         skip_updates=True)
