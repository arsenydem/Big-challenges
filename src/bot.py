import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot import types
from cfg import (
    BOT_TOKEN, 
    DB_PATH, 
    CSV_PATH, 
    API_KEY
)
from db import Database
from recsys import RecommendationSystem
from rag import RAGSystem
from states import UserStateManager
import numpy as np
import logging
import traceback
import requests
import io
import speech_recognition as sr
from texts import (
    WELCOME_MESSAGE,
    AGE_PROMPT,
    UPDATE_AGE_PROMPT,
    RECSYS,
    NEW_CHAT_PROMPT,
    CHOOSE_AGE_RANGE,
    CHOOSE_EXACT_AGE,
    AGE_SAVED,
    BACK_TO_RANGES,
    USE_BUTTONS,
    LIKE_MESSAGE,
    DISLIKE_MESSAGE,
    NEW_BOOK,
    BOOK_OK,
    ERROR,
    VOISE,
    WAIT,
    TRY,
    TRY_VOISE,
    PLZ,
    WAIT_RECSYS
)

# игнорирование предупреждений
import warnings
warnings.filterwarnings("ignore")

# логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

bot = AsyncTeleBot(BOT_TOKEN)

# основные компоненты бота
db = Database(DB_PATH)
recommender = RecommendationSystem(db, CSV_PATH)
rag = RAGSystem(CSV_PATH, API_KEY, db)
state_manager = UserStateManager()
recognizer = sr.Recognizer()

def create_feedback_keyboard(set_id):
    """Инлайн-клавиатура с кнопками 'лайк' и 'дизлайк'"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    like_button = types.InlineKeyboardButton("👍", callback_data=f"like_set_{set_id}")
    dislike_button = types.InlineKeyboardButton("👎", callback_data=f"dislike_set_{set_id}")
    keyboard.add(like_button, dislike_button)  
    return keyboard

def create_age_range_keyboard():
    """Инлайн-клавиатура для выбора диапазона возраста"""
    markup = types.InlineKeyboardMarkup(row_width=2)
    ranges = [
        ("8-17", (8, 17)), ("18-27", (18, 27)), ("28-37", (28, 37)),
        ("38-47", (38, 47)), ("48-57", (48, 57)), ("58-67", (58, 67)),
        ("68-77", (68, 77)), ("78-87", (78, 87)), ("88-97", (88, 97))
    ]
    buttons = [
        types.InlineKeyboardButton(
            label, callback_data=f"range_{range_tuple[0]}_{range_tuple[1]}"
        )
        for label, range_tuple in ranges
    ]
    for i in range(0, len(buttons), 2):
        if i + 1 < len(buttons):
            markup.add(buttons[i], buttons[i + 1])
        else:
            markup.add(buttons[i])
    return markup

def create_exact_age_keyboard(age_range):
    """Инлайн-клавиатура для выбора точного возраста"""
    markup = types.InlineKeyboardMarkup(row_width=2)
    start, end = age_range
    buttons = [
        types.InlineKeyboardButton(str(age), callback_data=f"age_{age}")
        for age in range(start, end + 1)
    ]
    for i in range(0, len(buttons), 2):
        if i + 1 < len(buttons):
            markup.add(buttons[i], buttons[i + 1])
        else:
            markup.add(buttons[i])
    back_button = types.InlineKeyboardButton(
        BACK_TO_RANGES, callback_data="back_to_ranges"
    )
    markup.add(back_button)
    return markup

def create_main_menu_keyboard():
    """Инлайн-клавиатура с кнопками 'Рекомендации', 'Новый чат' и 'Добавить книгу'."""
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton("Рекомендации", callback_data="forme"),
        types.InlineKeyboardButton("Новый чат", callback_data="newchat"),
        types.InlineKeyboardButton("Добавить книгу", callback_data="add_book"),
    ]
    markup.add(*buttons)
    return markup

@bot.message_handler(commands=["start"])
async def start_command(message):
    """/start (старт диалога с ботом)"""
    try:
        user_id = message.from_user.id
        language = message.from_user.language_code
        user_metadata = db.get_user_metadata(user_id)
        age = user_metadata.get("age") if user_metadata else None
        if age is None:
            state_manager.set_state(user_id, "waiting_for_age_range")
            if not user_metadata:
                db.save_user_profile(user_id, language=language)
            await bot.send_message(
                message.chat.id, AGE_PROMPT, reply_markup=create_age_range_keyboard()
            )
        else:
            state_manager.set_state(user_id, "idle")
            await bot.send_message(
                message.chat.id, WELCOME_MESSAGE, reply_markup=create_main_menu_keyboard(),
            )
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

@bot.message_handler(commands=["profile"])
async def profile_command(message):
    """/profile (обновление возраста)"""
    try:
        user_id = message.from_user.id
        language = message.from_user.language_code
        state_manager.set_state(user_id, "waiting_for_age_range")
        db.save_user_profile(user_id, language=language)
        await bot.send_message(
            message.chat.id, UPDATE_AGE_PROMPT, reply_markup=create_age_range_keyboard()
        )
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

@bot.message_handler(commands=["forme"])
async def forme_command(obj):
    """/forme (персональные рекомендации)"""
    try:
        if hasattr(obj, "chat"):
            chat_id = obj.chat.id
        else:
            chat_id = obj.message.chat.id
        user_id = obj.from_user.id
        user_metadata = db.get_user_metadata(user_id)
        age = user_metadata.get("age") if user_metadata else None
        if age is None:
            state_manager.set_state(user_id, "waiting_for_age_range")
            await bot.send_message(
                chat_id, AGE_PROMPT, reply_markup=create_age_range_keyboard()
            )
        else:
            state_manager.set_state(user_id, "idle")
            await bot.send_message(chat_id, WAIT_RECSYS)
            await bot.send_chat_action(chat_id, "typing")
            recommendations = recommender.recommend(user_id)
            response = RECSYS
            book_ids = []
            for idx, (book, book_id) in enumerate(recommendations, 1):
                response += f"{idx}. {book['Title']} by {book['Author']}\n"
                book_ids.append(book_id)
            set_id = db.save_recommendation_set(user_id, book_ids)
            await bot.send_message(
                chat_id, response, reply_markup=create_feedback_keyboard(set_id)
            )
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

@bot.message_handler(commands=["newchat"])
async def newchat_command(obj):
    """/newchat (начало нового чата с ботом)"""
    try:
        if hasattr(obj, "chat"):
            chat_id = obj.chat.id
            message_id = obj.message_id
        else:
            chat_id = obj.message.chat.id
            message_id = obj.message.message_id
        user_id = obj.from_user.id
        user_metadata = db.get_user_metadata(user_id)
        age = user_metadata.get("age") if user_metadata else None
        if age is None:
            state_manager.set_state(user_id, "waiting_for_age_range")
            await bot.send_message(
                chat_id, AGE_PROMPT, reply_markup=create_age_range_keyboard()
            )
        else:
            state_manager.set_state(user_id, "waiting_for_query")
            rag.clear_history(user_id)
            await bot.send_message(
                chat_id, NEW_CHAT_PROMPT, reply_to_message_id=message_id
            )
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

@bot.message_handler(commands=["newbook"])
async def newbook_command(message):
    """/newbook (добавление новой книги юзером)"""
    try:
        user_id = message.from_user.id
        await bot.send_message(message.chat.id, NEW_BOOK)
        state_manager.set_state(user_id, "waiting_for_new_book")
    except Exception as e:
        logger.error(
            f"{ERROR}: {str(e)}\n{traceback.format_exc()}"
        )

async def update_profile_task(user_id):
    """Фоновая задача для обновления юзер-эмбеддинга"""
    rag.update_profile_embedding(user_id)

@bot.message_handler(content_types=["text"])
async def handle_message(message):
    """Обрабатывает текстовые сообщения от юзера"""
    try:
        user_id = message.from_user.id
        state_info = state_manager.get_state(user_id)
        state = state_info["state"]
        user_metadata = db.get_user_metadata(user_id)
        age = user_metadata.get("age") if user_metadata else None

        if age is None:
            state_manager.set_state(user_id, "waiting_for_age_range")
            await bot.send_message(
                message.chat.id, AGE_PROMPT, reply_markup=create_age_range_keyboard()
            )
            return
        if state == "waiting_for_age_range":
            await bot.send_message(
                message.chat.id, CHOOSE_AGE_RANGE, reply_to_message_id=message.message_id,
            )
        elif state == "waiting_for_exact_age":
            await bot.send_message(
                message.chat.id, CHOOSE_EXACT_AGE, reply_to_message_id=message.message_id
            )
        elif state == "waiting_for_new_book":
            lines = message.text.split("\n")
            book_data = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in ["автор", "название", "описание", "жанр", "рейтинг"]:
                        book_data[key] = value

            required_fields = ["автор", "название", "описание", "жанр"]
            missing_fields = [field for field in required_fields if field not in book_data]
            if missing_fields:
                await bot.send_message(
                    message.chat.id,
                    f"{PLZ} {', '.join(missing_fields)}",
                )
                return

            new_book = {
                "Author": book_data["автор"],
                "Title": book_data["название"],
                "Description": book_data["описание"],
                "Genre": book_data["жанр"],
                "Rating": float(
                    book_data.get("рейтинг", 0)
                )
            }
            new_id, embedding = rag.add_book(new_book)

            recommender.add_book(new_id, embedding)

            rag.index.add(np.array([embedding]))

            await bot.send_message(message.chat.id, BOOK_OK)
            state_manager.set_state(user_id, "idle")
        elif state in ["waiting_for_query", "just_started", "idle"]:
            await bot.send_message(message.chat.id, WAIT)
            await bot.send_chat_action(message.chat.id, "typing")
            query = message.text
            response, books = rag.get_recommendation(user_id, query)
            book_ids = [book["id"] for book in books]
            set_id = db.save_recommendation_set(user_id, book_ids)
            await bot.send_message(
                message.chat.id, response, reply_markup=create_feedback_keyboard(set_id)
            )
            state_manager.set_state(user_id, "waiting_for_query")

            asyncio.create_task(update_profile_task(user_id))
        else:
            await bot.send_message(
                message.chat.id, USE_BUTTONS,
                reply_to_message_id=message.message_id, reply_markup=create_main_menu_keyboard(),
            )
    except Exception as e:
        logger.error(
            f"{ERROR}: {str(e)}\n{traceback.format_exc()}"
        )

@bot.message_handler(content_types=['voice'])
async def handle_voice_message(message):
    """Обрабатывает голосовой ввод"""
    try:
        user_id = message.from_user.id
        state_info = state_manager.get_state(user_id)
        state = state_info['state']
        user_metadata = db.get_user_metadata(user_id)
        age = user_metadata.get("age") if user_metadata else None

        if age is None:
            state_manager.set_state(user_id, "waiting_for_age_range")
            await bot.send_message(
                message.chat.id, AGE_PROMPT, reply_markup=create_age_range_keyboard()
            )
            return

        if state not in ["waiting_for_query", "just_started", "idle"]:
            await bot.send_message(
                message.chat.id, USE_BUTTONS,
                reply_to_message_id=message.message_id, reply_markup=create_main_menu_keyboard()
            )
            return

        file_info = await bot.get_file(message.voice.file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

        response = requests.get(file_url)
        if response.status_code != 200:
            logger.error(f"{ERROR}: {user_id}")
            await bot.send_message(message.chat.id, TRY_VOISE)
            return
        audio_buffer = io.BytesIO(response.content)
        audio_buffer.name = "voice.ogg"  
        audio_buffer.seek(0)

        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source)
        query = recognizer.recognize_google(audio_data, language="ru-RU")
        if not query:
            await bot.send_message(message.chat.id, TRY)
            return
        
        await bot.send_message(message.chat.id, f"Ваш запрос: {query}")
        await bot.send_message(message.chat.id, WAIT)
        await bot.send_chat_action(message.chat.id, "typing")
        response_text, books = rag.get_recommendation(user_id, query)
        book_ids = [book['id'] for book in books]
        set_id = db.save_recommendation_set(user_id, book_ids)
        await bot.send_message(
            message.chat.id, response_text, reply_markup=create_feedback_keyboard(set_id)
        )
        state_manager.set_state(user_id, "waiting_for_query")

        asyncio.create_task(update_profile_task(user_id))
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")
        await bot.send_message(message.chat.id, VOISE)

@bot.callback_query_handler(func=lambda call: True)
async def process_callback(call):
    """Callback-обработчики inline-кнопок"""
    try:
        user_id = call.from_user.id
        language = call.from_user.language_code
        data = call.data
        state_info = state_manager.get_state(user_id)
        state = state_info["state"]

        if data == "forme":
            await bot.edit_message_reply_markup(
                chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None
            )
            await forme_command(call)
        elif data == "newchat":
            await bot.edit_message_reply_markup(
                chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None
            )
            await newchat_command(call)
        elif data == "add_book":
            await bot.edit_message_reply_markup(
                chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None
            )
            await bot.send_message(
                call.message.chat.id, NEW_BOOK
            )
            state_manager.set_state(user_id, "waiting_for_new_book")
            await bot.answer_callback_query(call.id)
        elif data.startswith("range_") and state == "waiting_for_age_range":
            _, start, end = data.split("_")
            age_range = (int(start), int(end))
            state_manager.set_state(user_id, "waiting_for_exact_age", age_range)
            await bot.edit_message_text(
                chat_id=call.message.chat.id, message_id=call.message.message_id,
                text=f"Выбери точный возраст в диапазоне {start}-{end}:", reply_markup=create_exact_age_keyboard(age_range)
            )
            await bot.answer_callback_query(call.id)
        elif data.startswith("age_") and state == "waiting_for_exact_age":
            age = int(data.split("_")[1])
            db.save_user_profile(user_id, age=age)
            state_manager.set_state(user_id, "just_started")
            await bot.edit_message_text(
                chat_id=call.message.chat.id, message_id=call.message.message_id,
                text=AGE_SAVED.format(age=age), reply_markup=None,
            )
            await bot.answer_callback_query(call.id)
        elif data == "back_to_ranges" and state == "waiting_for_exact_age":
            state_manager.set_state(user_id, "waiting_for_age_range")
            await bot.edit_message_text(
                chat_id=call.message.chat.id, message_id=call.message.message_id,
                text="Выбери свой возраст:", reply_markup=create_age_range_keyboard()
            )
            await bot.answer_callback_query(call.id)
        elif data.startswith("like_set_") or data.startswith("dislike_set_"):
            await bot.edit_message_reply_markup(
                chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=None,
            )
            if action == "like":
                text = LIKE_MESSAGE
            elif action == "dislike":
                text = DISLIKE_MESSAGE
            await bot.answer_callback_query(call.id, text)
            action, set_id = data.split("_set_")
            set_id = int(set_id)
            book_ids = db.get_book_ids_from_set(set_id)
            for book_id in book_ids:
                db.add_interaction(user_id, book_id, action)
            if action == "like":
                rag.update_like_count(book_ids)
            elif action == "dislike":
                rag.update_dislike_count(book_ids)
            
            asyncio.create_task(update_profile_task(user_id))
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

async def main():
    """Поллинг бота"""
    try:
        await bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"{ERROR}: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())