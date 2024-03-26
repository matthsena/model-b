import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time
import threading

class LLMZeroShot:
    def __init__(self):
        self.lock = threading.Lock()  # Lock para serializar acesso ao DB
        self.chat = ChatOpenAI()

    # Função helper para executar consultas de forma segura
    def db_query(self, query, params=(), fetchone=False):
        try:
            with self.lock:  # Assegura que apenas uma thread acesse essa seção por vez
                conn = sqlite3.connect('database/llm_cache.db', check_same_thread=False)
                c = conn.cursor()
                c.execute(query, params)
                if fetchone:
                    result = c.fetchone()
                else:
                    result = c.fetchall()
                conn.close()
            return result
        except Exception as e:
            print(f"An error occurred in db_query: {e}")

    # Função helper para inserir dados no banco de forma segura
    def db_insert(self, query, params=()):
        try:
            with self.lock:  # Assegura que apenas uma thread insira dados por vez
                conn = sqlite3.connect('database/llm_cache.db', check_same_thread=False)
                c = conn.cursor()
                c.execute(query, params)
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"An error occurred in db_insert: {e}")

    def generate_text(self, preds1, preds2, panoptic1, panoptic2, ocr1, ocr2, path1, path2):
        try:
            start = time.time()

            panoptic1_text = ', '.join(panoptic1)
            panoptic2_text = ', '.join(panoptic2)
            ocr1_text = ', '.join(ocr1)
            ocr2_text = ', '.join(ocr2)

            user_content = f"Scene 1 could be either {preds1[0]} or {preds1[1]}, with objects {panoptic1_text} {f'texts: {ocr1_text}' if len(ocr1) > 0 else ''}. Scene 2 could be either {preds2[0]} or {preds2[0]}, with objects {panoptic2_text} {f'texts: {ocr2_text}' if len(ocr2) > 0 else ''}."

            row = self.db_query("SELECT answer FROM llm_cache WHERE question=?", (user_content,), fetchone=True)
            if row:
                return self.cache_hit(row[0], start)

            messages = [
                SystemMessage(
                    content="You are an AI assistant to help determine if two scenes are similar. Answer with yes if they are similar, otherwise, answer with no. Consider scenes similar if they contain the same objects in the same context and are in the same location."
                ),
                HumanMessage(
                    content=user_content
                ),
            ]

            answer = self.chat.invoke(messages)

            self.db_insert("INSERT INTO llm_cache VALUES (?, ?, ?, ?)", (user_content, answer.content.strip().lower(), path1, path2))

            return self.log_time_and_return_answer(answer.content.strip().lower(), start)
        except Exception as e:
            print(f"An error occurred in generate_text: {e}")

    def cache_hit(self, answer, start):
        try:
            # Logica para log de cache hit e calculo de tempo
            end = time.time()
            elapsed_time = end - start
            print(f"Cache hit: Resposta encontrada em {elapsed_time:.2f} segundos.")
            # Log de tempo aqui
            return answer
        except Exception as e:
            print(f"An error occurred in cache_hit: {e}")

    def log_time_and_return_answer(self, answer, start):
        try:
            # Logica para log de tempo de execução e retorno da resposta
            end = time.time()
            elapsed_time = end - start
            print(f"API response: Resposta obtida em {elapsed_time:.2f} segundos.")
            return answer
        except Exception as e:
            print(f"An error occurred in log_time_and_return_answer: {e}")

