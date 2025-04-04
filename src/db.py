import sqlite3
import numpy as np
import time
conn = None


class Database:
    def __init__(self, db_path):
        """Подключение к базе данных и создание таблицы"""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Таблицы в бд"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                book_id INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                book_id INTEGER,
                action TEXT,  
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_embedding BLOB,
                language TEXT,
                age INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_sets (
                set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                book_ids TEXT,  
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()
    
    def save_user_profile(self, user_id, embedding=None, language=None, age=None):
        """Сохраняет или обновляет профиль пользователя."""
        cursor = self.conn.cursor()
        embedding_blob = (
            embedding.tobytes() if isinstance(embedding, np.ndarray) else embedding
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO user_profiles (user_id, profile_embedding, language, age)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, embedding_blob, language, age),
        )
        self.conn.commit()

    def add_interaction(self, user_id, book_id, action):
        """Добавляет или обновляет взаимодействие пользователя с книгой"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT action FROM user_interactions 
            WHERE user_id = ? AND book_id = ?
            """,
            (user_id, book_id),
        )
        existing_action = cursor.fetchone()

        if existing_action:
            if existing_action[0] != action:
                cursor.execute(
                    """
                    UPDATE user_interactions 
                    SET action = ? 
                    WHERE user_id = ? AND book_id = ?
                    """,
                    (action, user_id, book_id),
                )
        else:
            cursor.execute(
                """
                INSERT INTO user_interactions (user_id, book_id, action)
                VALUES (?, ?, ?)
                """,
                (user_id, book_id, action),
            )

        self.conn.commit()


    def get_book_embeddings(self, book_ids):
        """Получает эмбеддинги книг по их id"""
        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in book_ids)
        cursor.execute(
            f"SELECT embedding FROM embeddings WHERE book_id IN ({placeholders})", book_ids
        )
        return [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
    def get_user_interactions(self, user_id):
        """Получает все взаимодействия пользователя."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT book_id, action 
            FROM user_interactions 
            WHERE user_id = ?
        """,
            (user_id,),
        )
        return [{"book_id": row[0], "action": row[1]} for row in cursor.fetchall()]
    
    def get_book_ids_from_set(self, set_id):
        """Получает список ID книг из набора рекомендаций."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT book_ids FROM recommendation_sets WHERE set_id = ?", (set_id,)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_interactions_for_book(self, book_id, action="view"):
        """Возвращает список взаимодействий для книги."""
        cursor = self.conn.cursor()
        query = """
            SELECT id, user_id, book_id, action, timestamp 
            FROM user_interactions 
            WHERE book_id = ? AND action = ?
        """
        cursor.execute(query, (book_id, action))
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "user_id": row[1],
                "book_id": row[2],
                "action": row[3],
                "timestamp": row[4],
            }
            for row in rows
        ]
        

    def save_interaction(self, user_id, book_id, action):
        """Сохраняет или обновляет взаимодействие пользователя с книгой"""
        timestamp = int(time.time())
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT action FROM user_interactions WHERE user_id = ? AND book_id = ?
        """,
            (user_id, book_id),
        )
        existing_interaction = cursor.fetchone()
        if existing_interaction:
            if existing_interaction[0] != action:
                cursor.execute(
                    """
                    UPDATE user_interactions 
                    SET action = ?, timestamp = ? 
                    WHERE user_id = ? AND book_id = ?
                """,
                    (action, timestamp, user_id, book_id),
                )
        else:
            cursor.execute(
                """
                INSERT INTO user_interactions (user_id, book_id, action, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (user_id, book_id, action, timestamp),
            )
        self.conn.commit()

    def get_user_metadata(self, user_id):
        """Получает метаданные пользователя из базы данных."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT language, age FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        result = cursor.fetchone()
        return (
            {"language": result[0], "age": result[1]}
            if result
            else {"language": None, "age": None}
        )

    def get_language_view_counts(self, book_id):
        """Возвращает количество просмотров книги, у пользователей с одним и тем же языком"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT up.language, COUNT(*) as view_count
            FROM user_interactions i
            JOIN user_profiles up ON i.user_id = up.user_id
            WHERE i.book_id = ? AND i.action = 'view'
            GROUP BY up.language
        """,
            (book_id,),
        )
        return dict(cursor.fetchall())

    def get_language_like_counts(self, book_id):
        """Возвращает количество лайков книги, у пользователей с одним и тем же языком"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT up.language, COUNT(*) as like_count
            FROM user_interactions i
            JOIN user_profiles up ON i.user_id = up.user_id
            WHERE i.book_id = ? AND i.action = 'like'
            GROUP BY up.language
        """,
            (book_id,),
        )
        return dict(cursor.fetchall())

    def save_recommendation_set(self, user_id, book_ids):
        """Сохраняет набор рекомендованных книг для пользователя и возвращает его id"""
        cursor = self.conn.cursor()
        book_ids_str = ",".join(map(str, book_ids))
        cursor.execute(
            "INSERT INTO recommendation_sets (user_id, book_ids) VALUES (?, ?)",
            (user_id, book_ids_str),
        )
        set_id = cursor.lastrowid
        self.conn.commit()
        return set_id

    def get_book_ids_from_set(self, set_id):
        """Извлекает список id книг из набора рекомендаций по его id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT book_ids FROM recommendation_sets WHERE set_id = ?", (set_id,)
        )
        result = cursor.fetchone()
        if result:
            return [int(book_id) for book_id in result[0].split(",")]
        return []
