import os
import numpy as np
import pandas as pd


class RecommendationSystem:
    def __init__(self, db, dataset_path):
        self.db = db
        self.dataset_path = dataset_path
        self.top_k = 5
        self.books_df = None
        self.books_df_mtime = None
        self.book_ids = []
        self.book_embeddings = None
        self.load_books()
        self.load_embeddings()

    def load_books(self):
        """Загружаем данные о книгах из CSV с кешированием по времени модификации файла."""
        current_mtime = os.path.getmtime(self.dataset_path)
        if self.books_df is None or current_mtime != self.books_df_mtime:
            self.books_df = pd.read_csv(self.dataset_path)
            self.books_df.fillna(
                {"mean_age": 30, "like_count": 0, "view_count": 0, "dislike_count": 0},
                inplace=True,
            )
            self.books_df_mtime = current_mtime
        return self.books_df

    def load_embeddings(self):
        """Загружаем эмбеддинги книг из базы данных."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT book_id, embedding FROM embeddings")
        embeddings_data = [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in cursor.fetchall()]
        if embeddings_data:
            self.book_ids, embeddings = zip(*embeddings_data)
            self.book_ids = list(self.book_ids)
            self.book_embeddings = np.stack(embeddings, axis=0)
        else:
            self.book_ids = []
            self.book_embeddings = np.array([])  

    def add_book(self, book_id, embedding):
        """Добавляет новую книгу в систему рекомендаций."""
        self.book_ids.append(book_id)
        if self.book_embeddings.size == 0:
            self.book_embeddings = embedding[np.newaxis, :]
        else:
            self.book_embeddings = np.vstack([self.book_embeddings, embedding])
        self.load_books()

    def recommend(self, user_id):
        # Получаем язык и возраст пользователя
        user_language = self.db.get_user_metadata(user_id).get("language", "ru")  
        user_age = self.db.get_user_metadata(user_id).get("age", 30) 

        books = self.load_books()

        user_embedding = self.db.get_user_metadata(user_id).get("profile_embedding")
        if user_embedding is not None and self.book_embeddings.size > 0:
            # Теплый старт
            user_norm = np.linalg.norm(user_embedding) or 1
            book_norms = np.linalg.norm(self.book_embeddings, axis=1, keepdims=True) or 1
            embedding_scores = np.dot(self.book_embeddings / book_norms, user_embedding / user_norm)

            filtered = books[books["id"].isin(self.book_ids)].copy()
            mean_ages = filtered["mean_age"].fillna(30).values.astype(np.float32)
            age_scores = np.exp(-np.abs(user_age - mean_ages) / 10)

            language_scores = []
            for bid in self.book_ids:
                view_count = self.db.get_language_view_counts(bid).get(user_language, 0)
                like_count = self.db.get_language_like_counts(bid).get(user_language, 0)
                language_score = (like_count * 2) + view_count
                language_scores.append(language_score * 0.1)

            final_scores = embedding_scores + np.array(language_scores) + (age_scores * 0.5)
            top_indices = np.argsort(-final_scores)[:self.top_k]

            return [
                (filtered.iloc[idx].to_dict(), self.book_ids[idx]) for idx in top_indices
            ]
        else:
            # Холодный старт
            books["score"] = (
                books["like_count"] * 0.7
                + books["view_count"] * 0.3
                - books["dislike_count"]
            )
            books["age_score"] = np.exp(-np.abs(user_age - books["mean_age"]) / 10)
            books["score"] += books["age_score"] * 0.5

            books["like_count_same_language"] = books["id"].apply(
                lambda x: self.db.get_language_like_counts(x).get(user_language, 0)
            )
            books["view_count_same_language"] = books["id"].apply(
                lambda x: self.db.get_language_view_counts(x).get(user_language, 0)
            )
            books["language_score"] = (books["like_count_same_language"] * 2) + books["view_count_same_language"]
            books["score"] += books["language_score"] * 0.1

            top_books = books.sort_values(by="score", ascending=False).head(self.top_k)
            return [(row.to_dict(), row["id"]) for _, row in top_books.iterrows()]