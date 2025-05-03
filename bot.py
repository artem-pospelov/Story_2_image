import os
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from story_image import gen_img  # Импортируем вашу функцию генерации

TOKEN = "7011351217:AAHArFPjVC13IlexGydcyn7eUsVk45SboBQ"
MAX_TEXT_LENGTH = 200  # Максимальная длина текста

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
📷 *Бот-генератор изображений*
Просто отправьте мне текстовое описание, и я создам изображение!

Пример:
_"Кот в шляпе, цифровая живопись"_

⚠ Ограничение: {max_len} символов
""".format(max_len=MAX_TEXT_LENGTH)
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    if len(user_text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(f"⚠ Слишком длинный текст. Максимум {MAX_TEXT_LENGTH} символов")
        return
    
    try:
        # Отправляем статус "печатает..."
        await update.message.chat.send_action(action="typing")
        
        # Генерируем изображение
        img = gen_img(user_text)
        
        # Сохраняем временно
        img_path = f"temp_{update.message.message_id}.jpg"
        img.save(img_path, "JPEG", quality=90)
        
        # Отправляем изображение
        with open(img_path, 'rb') as photo:
            await update.message.reply_photo(photo=InputFile(photo))
        
        # Удаляем временный файл
        os.remove(img_path)
        
    except Exception as e:
        await update.message.reply_text(f"⚠ Ошибка генерации: {str(e)}")

def main():
    app = Application.builder().token(TOKEN).build()
    
    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    
    # Обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()