import os
import faiss
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from deep_translator import GoogleTranslator
from cfg import INDEX_PATH
from texts import CONTEXT, PROMPT_SYSTEM


class RAGSystem:
    def __init__(self, csv_file, api_key, db, top_k=5, max_messages=10):
        """Инициализация RAG"""
        self.csv_file = csv_file
        self.db = db
        self.top_k = top_k
        self.api_key = api_key
        self.max_messages = max_messages
        self.model_cache_dir = "models"
        os.makedirs(self.model_cache_dir, exist_ok=True)

        self.books_df = pd.read_csv(csv_file)

        self.embed_model = self._load_sentence_transformer("all-MiniLM-L6-v2")
        self.cross_encoder = self._load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.vector_dim = 384
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            embeddings = self.embed_model.encode(
                self.books_df["Combined"].tolist(), convert_to_numpy=True
            )
            self._build_index(embeddings, INDEX_PATH)

        self.giga = GigaChat(
            credentials=self.api_key, verify_ssl_certs=False, temperature=0.7, top_p=0.35
        )
        self.memory = {}
        self.translator = GoogleTranslator(source="auto", target="en")

    def add_book(self, new_book):
        """Добавляет новую книгу в датасет и обновляет FAISS-индекс."""
        new_id = self.books_df["id"].max() + 1 if not self.books_df.empty else 1
        print(new_id)
        new_book["id"] = new_id

        self.books_df = pd.concat(
            [self.books_df, pd.DataFrame([new_book])], ignore_index=True
        )

        self.books_df.to_csv(self.csv_file, index=False)

        combined_text = f"{new_book['Title']} by {new_book['Author']}. Genre: {new_book.get('Genre', '')}. Description: {new_book.get('Description', '')}"
        
        embedding = self.embed_model.encode(combined_text, convert_to_numpy=True)
        self.index.add(np.array([embedding]))
        embedding = embedding.astype(np.float32)  

        return new_id, embedding

    def update_dislike_count(self, book_ids):
        """Обновляет dislike_count для книг в датасете"""
        for book_id in book_ids:
            self.books_df.loc[self.books_df["id"] == book_id, "dislike_count"] += 1
        self.books_df.to_csv(self.csv_file, index=False)

    def _load_sentence_transformer(self, model_name):
        """Загружает bi-encoder либо локально либо из интернета"""
        model_path = os.path.join(self.model_cache_dir, model_name)
        if os.path.exists(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            return SentenceTransformer(model_path)
        model = SentenceTransformer(model_name)
        model.save(model_path)
        return model

    def update_like_count(self, book_ids):
        """Обновляет like_count для книг в датасете"""
        for book_id in book_ids:
            self.books_df.loc[self.books_df["id"] == book_id, "like_count"] += 1
        self.books_df.to_csv(self.csv_file, index=False)

    def _load_cross_encoder(self, model_name):
        """Загружает cross-encoder либо локально либо из интернета"""
        model_path = os.path.join(self.model_cache_dir, model_name)
        if os.path.exists(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            return CrossEncoder(model_path)
        model = CrossEncoder(model_name)
        model.model.save_pretrained(model_path)
        model.tokenizer.save_pretrained(model_path)
        return model

    def _build_index(self, embeddings, index_file):
        """Создает HNSW FAISS-индекс (если его нет)"""
        self.index = faiss.IndexHNSWFlat(self.vector_dim, 32)
        self.index.hnsw.efConstruction = 128
        self.index.hnsw.efSearch = 128
        self.index.add(embeddings)
        faiss.write_index(self.index, index_file)

    def _get_or_create_memory(self, user_id):
        """Создает историю сообщений пользователя (если её нет)"""
        if user_id not in self.memory:
            self.memory[user_id] = [
                SystemMessage(
                    content=PROMPT_SYSTEM
                )
            ]
        return self.memory[user_id]

    def translate_to_english(self, text):
        """Перевод на английский"""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return text

    def search_books(self, user_id, query, initial_k=100):
        """Поиск книг по запросу"""
        query_en = self.translate_to_english(query)
        query_embedding = self.embed_model.encode(query_en, convert_to_numpy=True)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        _, indices = self.index.search(query_embedding, initial_k)
        results = self.books_df.iloc[indices[0]].to_dict(orient="records")

        book_ids = [book["id"] for book in results]

        self.update_view_count(book_ids)

        self.update_mean_age_for_books(user_id, book_ids)

        return results

    def update_view_count(self, book_ids):
        """Увеличивает view_count в датасете"""
        self.books_df.loc[self.books_df["id"].isin(book_ids), "view_count"] += 1
        self.books_df.to_csv(self.csv_file, index=False)

    def update_mean_age_for_books(self, user_id, book_ids):
        """Обновляет mean_age"""
        user_metadata = self.db.get_user_metadata(user_id)
        user_age = user_metadata.get("age")

        if user_age is None:
            return  

        for book_id in book_ids:
            self.db.save_interaction(user_id, book_id, "view")

            interactions = self.db.get_interactions_for_book(book_id, action="view")
            ages = [
                self.db.get_user_metadata(i["user_id"]).get("age") for i in interactions
            ]
            ages = [age for age in ages if age is not None]

            new_mean_age = np.mean(ages) if ages else user_age

            self.books_df.loc[self.books_df["id"] == book_id, "mean_age"] = new_mean_age

        self.books_df.to_csv(self.csv_file, index=False)

    def rerank_with_cross_encoder(self, query, books):
        """Reranking"""
        pairs = [
            (
                query,
                f"{book['Title']} by {book['Author']}. Genre: {book.get('Genre', '')}. Description: {book.get('Description', '')}",
            )
            for book in books
        ]
        scores = self.cross_encoder.predict(pairs)
        sorted_books = [
            book
            for _, book in sorted(zip(scores, books), key=lambda x: x[0], reverse=True)
        ]
        return sorted_books[: self.top_k]

    def update_history(self, user_id, role, message):
        """Обновление истории"""
        memory = self._get_or_create_memory(user_id)
        if role == "user":
            memory.append(HumanMessage(content=message))
        else:
            memory.append(AIMessage(content=message))
        if len(memory) > self.max_messages:
            memory[1:] = memory[-self.max_messages + 1 :]

    def clear_history(self, user_id):
        """Удаляет историю"""
        if user_id in self.memory:
            del self.memory[user_id]

    def build_prompt(self, user_id, user_query, books_context):
        """Промпт"""
        self.update_history(user_id, "user", user_query)
        prompt = f"\n{CONTEXT}\n"
        for idx, book in enumerate(books_context, 1):
            prompt += f"{idx}. {book['Title']}. Автор: {book['Author']})\n"
        self.update_history(user_id, "user", prompt)

    def update_profile_embedding(self, user_id):
        """Обновляет эмбеддинг профиля пользователя на основе его взаимодействий с книгами."""
        interactions = self.db.get_user_interactions(user_id)
        if not interactions:
            return

        action_weights = {"like": 2, "view": 1, "dislike": -2}
        book_ids, weights = [], []

        for interaction in interactions:
            book_ids.append(interaction["book_id"])
            weights.append(action_weights.get(interaction["action"], 0.0))

        book_embeddings = self.db.get_book_embeddings(book_ids)

        valid_data = [(emb, w) for emb, w in zip(book_embeddings, weights) if emb is not None]
        if not valid_data:
            return

        embeddings_array = np.array([emb for emb, _ in valid_data])
        weights_array = np.array([w for _, w in valid_data])

        mean_embedding = np.sum(embeddings_array * weights_array[:, np.newaxis], axis=0) / np.sum(np.abs(weights_array))

        user_metadata = self.db.get_user_metadata(user_id)
        self.db.save_user_profile(user_id, mean_embedding, user_metadata.get("language"), user_metadata.get("age"))

    def get_recommendation(self, user_id, user_query):
        """Получение ответа"""
        books_context = self.search_books(user_id, user_query, initial_k=100)
        books_context = self.rerank_with_cross_encoder(user_query, books_context)

        book_ids = [book["id"] for book in books_context]
        for book_id in book_ids:
            self.db.add_interaction(user_id, book_id, "view")

        self.build_prompt(
            user_id, user_query, books_context
        ) 

        memory = self._get_or_create_memory(user_id)
        response = self.giga.invoke(memory)
        self.update_history(user_id, "assistant", response.content)

        return (
            self.clean_text(response.content),
            books_context,
        ) 

    def clean_text(self, text):
        """Очистка текста от лишних символов."""
        return re.sub(r"\s+", " ", re.sub(r"#+\s*|\*+", "", text)).strip()
