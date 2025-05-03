import io
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor
from story_image import *

# Замените на ваш токен
API_TOKEN = '7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ'

# Инициализация бота
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class Form(StatesGroup):
    waiting_for_text = State()

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.answer("Привет! Отправь мне историю, и я сгенерирую для тебя иллюстрации к ней.")
    await Form.waiting_for_text.set()

@dp.message_handler(state=Form.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    user_text = message.text
    
    # Получаем данные от функции gen_img
    best_frames, best_clips, best_lpips, story_ls = gen_img(user_text)
    
    # Проверяем, что все списки одинаковой длины
    if not (len(best_frames) == len(best_clips) == len(best_lpips) == len(story_ls)):
        await message.answer("Ошибка: получены списки разной длины")
        await state.finish()
        return
    
    # Отправляем пользователю все сгенерированные сообщения
    for i in range(len(story_ls)):
        # Формируем текст сообщения с метриками
        msg_text = f"{story_ls[i]}\n\nМетрики:\nCLIP: {best_clips[i]}\nLPIPS: {best_lpips[i]}"
        
        # Конвертируем изображение в байты
        img_byte_arr = io.BytesIO()
        best_frames[i].save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Отправляем фото с подписью
        await message.answer_photo(types.InputFile(img_byte_arr), caption=msg_text)
    
    await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
