import os
import sys
import sqlite3
import pyodbc
import time
from datetime import datetime

# env and api keys
from dotenv import load_dotenv

# PyQt5 imports
from PyQt5.QtWidgets import(QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QTextEdit, QLineEdit, QLabel, QGroupBox, QRadioButton,
                            QFileDialog, QSplitter, QMessageBox, QStatusBar, QDialog, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# langchain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# llm and embeddings imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 1. configuration and services

load_dotenv()

class LLMService:
    """Manages the creation of LLM and embedding models"""
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def get_llm(self, provider: str):
        if provider == "OpenAI":
            if not self.openai_api_key: raise ValueError("OpenAI API key not found.")
            return ChatOpenAI(model="gpt-4o", openai_api_key=self.openai_api_key)
        if provider == "Groq":
            if not self.groq_api_key: raise ValueError("GROQ API key not found.")
            return ChatGroq(model="Llama3-8b-8192", groq_api_key=self.groq_api_key)
        if provider == "Ollama":
            return ChatOllama(model="Llama3")
        raise ValueError(f"Unknown LLM provider: {provider}")

    def get_embedding_model(self, provider: str):
        if provider in ["OpenAI", "Groq"]:
            if not self.openai_api_key: raise ValueError("OpenAI key not found for embeddings.")
            return OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        if provider == "Ollama":
            return OllamaEmbeddings(model="nomic-embed-text")
        raise ValueError(f"Unknown provider for embeddings: {provider}")

class DatabaseLogger:
    """Handles loggings of Q&A interactions to either SQLite or SQL Server"""
    def __init__(self):
        self.conn = None
        self.db_type = None

    def connect(self, db_type: str, **kwargs):
        self.db_type = db_type
        try:
            if self.db_type == "SQLite":
                self.conn = sqlite3.connect('rag_log.db', check_same_thread=False)
            elif self.db_type == "SQL Server":
                # ensure the microsoft odbc driver is installed on your system
                conn_str = (
                   f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                   f'SERVER={kwargs.get("server")};'
                   f'DATABASE={kwargs.get("database")};'
                   f'UID={kwargs.get("username")};'
                   f'PWD={kwargs.get("password")}'
                )
                self.conn = pyodbc.connect(conn_str)

            self.create_log_table()
            return True, f"Successfully connected to {self.db_type}."
        except Exception as e:
            self.conn = None
            return False, f"Failed to connect to {self.db_type}."

    def create_log_table(self):
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            if self.db_type == "SQL Server":
                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='qa_logs' and xtype='U')
                    CREATE TABLE qa_logs (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        llm_provider NVARCHAR(50) NOT NULL,
                        question NVARCHAR(MAX) NOT NULL,
                        answer NVARCHAR(MAX) NOT NULL,
                        response_time FLOAT NOT NULL
                    );
                """)
            else:   # sqlite
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS qa_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        llm_provider TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        response_time REAL NOT NULL
                    );
                """)
            self.conn.commit()
        except Exception as e:
            print(f"Error creating log table: {e}")

    def log_interaction(self, provider, question, answer, response_time):
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO qa_logs (timestamp, llm_provider, question, answer, response_time) VALUES (?, ?, ?, ?, ?)",
                (datetime.now(), provider, question, answer, response_time)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error logging to database: {e}")

# 2. dialog for SQL Server credentials

class SQLServerDialog(QDialog):
    """A dialog window to securely get SQL Server credentials from the user."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SQL Server Connection")
        self.layout = QFormLayout(self)

        self.server = QLineEdit(self)
        self.database = QLineEdit(self)
        self.username = QLineEdit(self)
        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)

        self.layout.addRow("Server:", self.server)
        self.layout.addRow("Database:", self.database)
        self.layout.addRow("Username:", self.username)
        self.layout.addRow("Password:", self.password)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        cancel_button = QPushButton("Cancel", self)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        self.layout.addRow(buttons)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def get_details(self):
        return {
            "server": self.server.text(),
            "database": self.database.text(),
            "username": self.username.text(),
            "password": self.password.text(),
        }

# 3. RAG worker threads

class IndexingWorker(QThread):
    """Worker thread for handling the document indexing process."""
    finished = pyqtSignal(object, str)

    def __init__(self, doc_path, embedding_model, save_path):
        super().__init__()
        self.doc_path = doc_path
        self.embedding_model = embedding_model
        self.save_path = save_path

    def run(self):
        try:
            loader = PyPDFDirectoryLoader(self.doc_path)
            docs = loader.load()
            if not docs:
                self.finished.emit(None, "No PDF files found in the selected directory.")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            vector_store = FAISS.from_documents(final_documents, self.embedding_model)
            vector_store.save_local(self.save_path)

            self.finished.emit(vector_store, f"✅ Indexing complete and saved. {len(final_documents)}")
        except Exception as e:
            self.finished.emit(None, f"❌ Error during indexing. {e}")

class QueryWorker(QThread):
    """Worker thread for handling RAG queries."""
    finished = pyqtSignal(dict, float)

    def __init__(self, chain, query):
        super().__init__()
        self.chain = chain
        self.query = query

    def run(self):
        try:
            start_time = time.time()
            response = self.chain.invoke({'input': self.query})
            duration = time.time() - start_time
            self.finished.emit(response, duration)
        except Exception as e:
            self.finished.emit({'answer': f"An error occurred: {e}", 'context': []}, 0)

# 4. main application ui

class RAGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced RAG document Q&A (v2)")
        self.setGeometry(100, 100, 1200, 800)

        self.llm_service = LLMService()
        self.db_logger = DatabaseLogger()
        self.vector_store = None
        self.rag_chain = None
        self.doc_path = "research_papers"
        self.last_query = ""

        self.indexing_worker = None
        self.query_worker = None

        self.initUI()
        self.connect_to_db()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)

        # left panel controls

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # llm selection

        llm_group = QGroupBox("🤖 LLM Provider")
        llm_layout = QVBoxLayout()
        self.rb_openai = QRadioButton("OpenAI (gpt-4o)")
        self.rb_groq = QRadioButton("Groq (llama-3.1-8b-instant)")
        self.rb_ollama = QRadioButton("Ollama (local)")
        self.rb_groq.setChecked(True)
        llm_layout.addWidget(self.rb_openai)
        llm_layout.addWidget(self.rb_groq)
        llm_layout.addWidget(self.rb_ollama)
        llm_group.setLayout(llm_layout)
        left_layout.addWidget(llm_group)

        # database selection

        db_group = QGroupBox("🗄️ Database Logger")
        db_layout = QVBoxLayout()
        self.rb_sqlite = QRadioButton("SQLite (local)")
        self.rb_sqlserver = QRadioButton("SQL Server")
        self.rb_sqlite.setChecked(True)
        db_layout.addWidget(self.rb_sqlite)
        db_layout.addWidget(self.rb_sqlserver)
        db_group.setLayout(db_layout)
        self.rb_sqlite.toggled.connect(self.connect_to_db)
        self.rb_sqlserver.toggled.connect(self.connect_to_db)
        left_layout.addWidget(db_group)

        # Document management

        doc_group = QGroupBox("📄 Document & Vector Store")
        doc_layout = QVBoxLayout()
        self.doc_path_label = QLabel(f"Source: {self.doc_path}")
        self.doc_path_label.setWordWrap(True)
        self.select_folder_btn = QPushButton("Select PDF Folder")
        self.create_index_btn = QPushButton("Create & Save New Index")
        self.load_index_btn = QPushButton("Load Existing Index")
        doc_layout.addWidget(self.doc_path_label)
        doc_layout.addWidget(self.select_folder_btn)
        doc_layout.addWidget(self.create_index_btn)
        doc_layout.addWidget(self.load_index_btn)
        doc_group.setLayout(doc_layout)
        left_layout.addWidget(doc_group)

        splitter.addWidget(left_panel)

        # Right panel (Q&A)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        context_group = QGroupBox("Retrieved Context")
        context_layout = QVBoxLayout()
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setFixedHeight(200)
        context_layout.addWidget(self.context_display)
        context_group.setLayout(context_layout)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Please load an index to begin...")
        self.user_input.setEnabled(False)
        self.send_btn = QPushButton("Send")
        self.send_btn.setEnabled(False)
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_btn)

        right_layout.addWidget(self.chat_display)
        right_layout.addWidget(context_group)
        right_layout.addLayout(input_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])
        self.main_layout.addWidget(splitter)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Select a document folder and create or load a vector index.")

        # connections

        self.select_folder_btn.clicked.connect(self.select_document_folder)
        self.create_index_btn.clicked.connect(self.create_vector_index)
        self.load_index_btn.clicked.connect(self.load_vector_index)
        self.send_btn.clicked.connect(self.handle_query)
        self.user_input.returnPressed.connect(self.handle_query)

    def get_selected_provider(self):
        if self.rb_openai.isChecked(): return "OpenAI"
        if self.rb_groq.isChecked(): return "Groq"
        if self.rb_ollama.isChecked(): return "Ollama"
        return "Groq"

    def connect_to_db(self):
        if self.rb_sqlite.isChecked():
            success, message = self.db_logger.connect("SQLite")
            self.statusBar.showMessage(message)
            if not success: QMessageBox.critical(self, "Database Error", message)
        elif self.rb_sqlserver.isChecked():
            dialog = SQLServerDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                details = dialog.get_details()
                success, message = self.db_logger.connect("SQL Server", **details)
                self.statusBar.showMessage(message)
                if not success: QMessageBox.critical(self, "Database Error", message)
            else:
                self.rb_sqlite.setChecked(True)   # revert selection if canceled

    def select_document_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with PDFs.")
        if folder:
            self.doc_path = folder
            self.doc_path_label.setText(f"Source: {self.doc_path}")
            self.statusBar.showMessage("New folder selected. Please create or load an index.")
            self._reset_chat_state()

    def create_vector_index(self):
        save_path = QFileDialog.getExistingDirectory(self, "Select folder to save index.")
        if not save_path: return

        provider = self.get_selected_provider()
        try:
            embedding_model = self.llm_service.get_embedding_model(provider)
        except ValueError as e:
            QMessageBox.critical(self, "API Key error.", str(e)); return

        self.statusBar.showMessage(f"⏳ Creating new vector index...")
        self.create_index_btn.setEnabled(False)
        self.load_index_btn.setEnabled(False)

        self.indexing_worker = IndexingWorker(self.doc_path, embedding_model, save_path)
        self.indexing_worker.finished.connect(self.on_indexing_finished)
        self.indexing_worker.start()

    def load_vector_index(self):
        load_path = QFileDialog.getExistingDirectory(self, "Select folder with saved index.")
        if not load_path: return

        if not os.path.exists(os.path.join(load_path, "index.faiss")):
            QMessageBox.critical(self, "Error", "Invalid index folder. 'index.faiss' not found.")
            return

        provider = self.get_selected_provider()
        try:
            embedding_model = self.llm_service.get_embedding_model(provider)
            self.statusBar.showMessage(f"⏳ Loading index with {provider} embeddings...")

            # FAISS.load_local can be slow, so it is best practice to do it in a thread
            # for simplicity here we do it on the main thread but show a status message
            self.vector_store = FAISS.load_local(load_path, embedding_model, allow_dangerous_deserialization=True)
            self.setup_rag_chain()
            self.statusBar.showMessage("✅ Index loaded successfully. You can now ask questions.")
        except Exception as e:
            QMessageBox.critical(self, "Error loading index", f"Failed to load index. Ensure you are using the correct embedding model. {e}")
            self._reset_chat_state()
            self.statusBar.showMessage("❌ Failed to load index.")

    def on_indexing_finished(self, vector_store, status_message):
        self.statusBar.showMessage(status_message)
        self.create_index_btn.setEnabled(True)
        self.load_index_btn.setEnabled(True)
        if vector_store:
            self.vector_store = vector_store
            self.setup_rag_chain()
        else:
            QMessageBox.warning(self, "Indexing failed", status_message)

    def setup_rag_chain(self):
        provider = self.get_selected_provider()
        try:
            llm = self.llm_service.get_llm(provider)
            prompt = ChatPromptTemplate.from_template(
                "Answer the questions based on the provided context only. \n"
                "Provide the most accurate and detailed response. \n"
                "<context>{context}</context>\nQuestion: {input}"
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = self.vector_store.as_retriever()
            self.rag_chain = create_retrieval_chain(retriever, document_chain)

            self.user_input.setPlaceholderText("Ask a question about the documents...")
            self.user_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.statusBar.showMessage(f"✅ RAG chain is ready with {provider}. Ask away!")
        except ValueError as e:
            QMessageBox.critical(self, "API Key error", str(e))
            self.statusBar.showMessage(f"❌ Error: {e}")

    def handle_query(self):
        user_query = self.user_input.text().strip()
        if not user_query or not self.rag_chain: return

        self.last_query = user_query
        self.chat_display.append(f"<b style='color: #00008B;'>You:</b> {user_query}")
        self.user_input.clear()
        self.statusBar.showMessage(f"🤔 Thinking with {self.get_selected_provider()}...")
        self.send_btn.setEnabled(False)

        self.query_worker = QueryWorker(self.rag_chain, user_query)
        self.query_worker.finished.connect(self.on_query_finished)
        self.query_worker.start()

    def on_query_finished(self, response, duration):
        answer = response.get('answer', 'Sorry, I could not find an answer.')
        self.chat_display.append(f"<b style='color: #006400;'>Assistant:</b> {answer}")

        context_text = "\n\n".join([f"--- CONTEXT {i+1} ---\n{doc.page_content}" for i, doc in enumerate(response.get('context', []))])
        self.context_display.setText(context_text)

        self.db_logger.log_interaction(
            provider = self.get_selected_provider(),
            question = self.last_query,
            answer = answer,
            response_time = duration
        )
        self.statusBar.showMessage(f"Ready. Response received in {duration:.2f} seconds.")
        self.send_btn.setEnabled(True)

    def _reset_chat_state(self):
        self.rag_chain = None
        self.chat_display.clear()
        self.context_display.clear()
        self.user_input.setPlaceholderText("Please load an index to begin...")
        self.user_input.setEnabled(False)
        self.send_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.db_logger.conn: self.db_logger.conn.close()
        event.accept()

# 5. application entry point

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RAGApp()
    main_window.show()
    sys.exit(app.exec())
