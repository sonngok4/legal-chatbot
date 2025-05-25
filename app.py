import os
import re
import sqlite3
from datetime import datetime
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logging_config import setup_logging

app = Flask(__name__)
CORS(app)  # Cho phép CORS để Flutter có thể gọi API

# Setup logging
logger = setup_logging()


class LegalChatbot:
    def __init__(self):
        logger.info("Initializing LegalChatbot")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=1000, stop_words=None  # Unigram và bigram
        )
        self.db_path = "legal_database.db"
        self.load_data()
        self.build_search_index()
        logger.info("LegalChatbot initialization completed")

    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    def load_data(self):
        """Load data from SQLite database"""
        try:
            logger.info("Loading data from database")
            conn = self.get_db_connection()
            self.violations = pd.read_sql_query("SELECT * FROM violations", conn)
            self.legal_documents = pd.read_sql_query(
                "SELECT * FROM legal_documents", conn
            )
            conn.close()
            logger.info(
                f"Loaded {len(self.violations)} violations and {len(self.legal_documents)} legal documents"
            )

            # Tạo corpus để tìm kiếm
            self.corpus = []
            for _, violation in self.violations.iterrows():
                text = f"{violation['description']} {violation['keywords']} {violation['vehicle_type']} {violation['violation_type']}"
                self.corpus.append(text.lower())
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def build_search_index(self):
        """Xây dựng vector search index"""
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            print("Đã tạo search index thành công")
        else:
            print("Không có dữ liệu để tạo search index")

    def preprocess_text(self, text):
        """Tiền xử lý văn bản tiếng Việt"""
        text = text.lower()

        # Chuẩn hóa từ khóa
        replacements = {
            "xe gắn máy": "xe máy",
            "moto": "xe máy",
            "xe mô tô": "xe máy",
            "ô tô con": "ô tô",
            "xe hơi": "ô tô",
            "xe con": "ô tô",
            "rượu bia": "nồng độ cồn",
            "say xỉn": "nồng độ cồn",
            "uống rượu": "nồng độ cồn",
            "vượt đèn đỏ": "đèn đỏ",
            "chạy qua đèn đỏ": "đèn đỏ",
            "không đội mũ bảo hiểm": "mũ bảo hiểm",
            "không có mũ bảo hiểm": "mũ bảo hiểm",
            "chạy quá tốc độ": "tốc độ",
            "vượt tốc độ": "tốc độ",
            "bằng lái": "giấy phép lái xe",
            "giấy phép": "giấy phép lái xe",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def extract_entities(self, text):
        """Trích xuất thông tin từ câu hỏi"""
        text = self.preprocess_text(text)

        entities = {
            "vehicle_type": None,
            "violation_types": [],
            "speed": None,
            "alcohol_level": None,
        }

        # Nhận diện loại phương tiện
        if any(word in text for word in ["xe máy", "moto"]):
            entities["vehicle_type"] = "xe máy"
        elif any(word in text for word in ["ô tô", "xe hơi", "xe con"]):
            entities["vehicle_type"] = "ô tô"

        # Nhận diện loại vi phạm (có thể nhiều loại)
        if any(word in text for word in ["tốc độ", "chạy nhanh", "km/h", "quá tốc độ"]):
            entities["violation_types"].append("tốc độ")
            # Trích xuất tốc độ
            speed_match = re.search(r"(\d+)\s*km/h", text)
            if speed_match:
                entities["speed"] = int(speed_match.group(1))

        if any(word in text for word in ["cồn", "rượu", "bia", "say"]):
            entities["violation_types"].append("nồng độ cồn")

        if any(word in text for word in ["đèn đỏ", "vượt đèn", "đèn tín hiệu"]):
            entities["violation_types"].append("đèn đỏ")

        if any(word in text for word in ["mũ bảo hiểm", "không đội", "mũ"]):
            entities["violation_types"].append("mũ bảo hiểm")

        if any(word in text for word in ["giấy phép", "bằng lái", "giấy tờ"]):
            entities["violation_types"].append("giấy tờ")

        return entities

    def search_violations(self, query, top_k=5):
        """Tìm kiếm vi phạm phù hợp"""
        if not self.corpus:
            return []

        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])

        # Tính cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Lấy top kết quả
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        entities = self.extract_entities(query)

        for idx in top_indices:
            if similarities[idx] > 0.05:  # Threshold thấp hơn
                violation = self.violations.iloc[idx].to_dict()
                violation["confidence"] = float(similarities[idx])

                # Bonus điểm nếu match vehicle type
                if (
                    entities["vehicle_type"]
                    and entities["vehicle_type"] in violation["vehicle_type"]
                ):
                    violation["confidence"] += 0.2
                elif violation["vehicle_type"] == "tất cả":
                    violation["confidence"] += 0.1

                # Bonus điểm nếu match violation type
                if entities["violation_types"]:
                    for vtype in entities["violation_types"]:
                        if vtype in violation["violation_type"]:
                            violation["confidence"] += 0.3
                            break

                results.append(violation)

        # Sắp xếp lại theo confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:3]  # Chỉ lấy top 3

    def generate_response(self, query):
        """Sinh câu trả lời"""
        try:
            logger.info(f"Processing query: {query}")
            if not query.strip():
                logger.warning("Empty query received")
                return {"answer": "Vui lòng nhập câu hỏi của bạn.", "confidence": 0.0}

            violations = self.search_violations(query)
            entities = self.extract_entities(query)

            if not violations:
                logger.info("No violations found for query")
                suggestion = "Bạn có thể hỏi về:\n"
                suggestion += (
                    "- Vi phạm tốc độ (VD: 'xe máy chạy 80km/h bị phạt bao nhiêu?')\n"
                )
                suggestion += (
                    "- Vi phạm nồng độ cồn (VD: 'uống rượu lái xe máy bị phạt gì?')\n"
                )
                suggestion += "- Vượt đèn đỏ (VD: 'ô tô vượt đèn đỏ phạt bao nhiêu?')\n"
                suggestion += "- Không đội mũ bảo hiểm\n"
                suggestion += "- Không mang giấy phép lái xe"

                return {
                    "answer": f'Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi "{query}".\n\n{suggestion}',
                    "confidence": 0.0,
                }

            # Xây dựng câu trả lời
            response = f'**Thông tin về câu hỏi: "{query}"**\n\n'

            for i, violation in enumerate(violations, 1):
                confidence_text = (
                    f" (độ chính xác: {violation['confidence']:.1%})"
                    if violation["confidence"] < 0.8
                    else ""
                )

                response += f"**{i}. {violation['description']}{confidence_text}**\n"
                response += f"📋 **Áp dụng cho:** {violation['vehicle_type']}\n"
                response += f"💰 **Mức phạt:** {violation['fine_amount']}\n"

                if violation["additional_penalty"]:
                    response += (
                        f"⚠️ **Hình phạt bổ sung:** {violation['additional_penalty']}\n"
                    )

                response += f"📖 **Căn cứ pháp lý:** {violation['legal_reference']}\n\n"

            response += "---\n"
            response += "*💡 Lưu ý: Thông tin này chỉ mang tính chất tham khảo. Trong trường hợp cụ thể, vui lòng tham khảo ý kiến của cơ quan có thẩm quyền hoặc luật sư.*"

            return {
                "answer": response,
                "confidence": violations[0]["confidence"] if violations else 0.0,
                "violations_found": len(violations),
                "entities": entities,
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                "confidence": 0.0,
            }

    @lru_cache(maxsize=100)
    def cached_search(self, query):
        return self.search_violations(query)

    def learn_from_feedback(self, query, is_helpful):
        """Log feedback để cải thiện"""
        with open("feedback.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()}: {query} | Helpful: {is_helpful}\n")

    # Add new methods for data management
    def add_violation(self, violation_data):
        """Add new violation"""
        try:
            logger.info(f"Adding new violation: {violation_data['description']}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
            INSERT INTO violations (
                violation_type, description, vehicle_type, fine_amount,
                additional_penalty, legal_reference, keywords, document_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    violation_data["violation_type"],
                    violation_data["description"],
                    violation_data["vehicle_type"],
                    violation_data["fine_amount"],
                    violation_data.get("additional_penalty", ""),
                    violation_data["legal_reference"],
                    violation_data["keywords"],
                    violation_data.get("document_id"),
                ),
            )

            violation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.load_data()
            logger.info(f"Successfully added violation with ID: {violation_id}")
            return violation_id
        except Exception as e:
            logger.error(f"Error adding violation: {str(e)}")
            raise

    def update_violation(self, violation_id, violation_data):
        """Update existing violation"""
        try:
            logger.info(f"Updating violation ID: {violation_id}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            update_fields = []
            values = []
            for key, value in violation_data.items():
                if key in [
                    "violation_type",
                    "description",
                    "vehicle_type",
                    "fine_amount",
                    "additional_penalty",
                    "legal_reference",
                    "keywords",
                    "document_id",
                ]:
                    update_fields.append(f"{key} = ?")
                    values.append(value)

            if update_fields:
                query = f"UPDATE violations SET {', '.join(update_fields)} WHERE id = ?"
                values.append(violation_id)
                cursor.execute(query, values)
                conn.commit()
                success = cursor.rowcount > 0
                logger.info(f"Violation update {'successful' if success else 'failed'}")
            else:
                success = False
                logger.warning("No valid fields to update")

            conn.close()
            self.load_data()
            return success
        except Exception as e:
            logger.error(f"Error updating violation: {str(e)}")
            raise

    def delete_violation(self, violation_id):
        """Delete violation"""
        try:
            logger.info(f"Deleting violation ID: {violation_id}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()

            self.load_data()
            logger.info(f"Violation deletion {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Error deleting violation: {str(e)}")
            raise

    def add_legal_document(self, document_data):
        """Add new legal document"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO legal_documents (title, code, content, effective_date, status)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                document_data["title"],
                document_data["code"],
                document_data["content"],
                document_data["effective_date"],
                document_data.get("status", "active"),
            ),
        )

        conn.commit()
        conn.close()
        self.load_data()  # Reload data
        return cursor.lastrowid

    def update_legal_document(self, doc_id, document_data):
        """Update existing legal document"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        update_fields = []
        values = []
        for key, value in document_data.items():
            if key in ["title", "code", "content", "effective_date", "status"]:
                update_fields.append(f"{key} = ?")
                values.append(value)

        if update_fields:
            query = (
                f"UPDATE legal_documents SET {', '.join(update_fields)} WHERE id = ?"
            )
            values.append(doc_id)
            cursor.execute(query, values)
            conn.commit()

        conn.close()
        self.load_data()  # Reload data
        return cursor.rowcount > 0

    def delete_legal_document(self, doc_id):
        """Delete legal document"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # First delete related violations
        cursor.execute("DELETE FROM violations WHERE document_id = ?", (doc_id,))
        # Then delete the document
        cursor.execute("DELETE FROM legal_documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

        self.load_data()  # Reload data
        return cursor.rowcount > 0


# Khởi tạo chatbot
print("Đang khởi tạo chatbot...")
chatbot = LegalChatbot()
print("Chatbot đã sẵn sàng!")


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Legal Chatbot API đang hoạt động!",
            "endpoints": {
                "chat": "/chat (POST)",
                "webhook": "/webhook (POST - cho Dialogflow)",
            },
        }
    )


@app.route("/webhook", methods=["POST"])
def webhook():
    """Webhook cho Dialogflow"""
    req = request.get_json()

    # Lấy query từ Dialogflow
    query_text = req.get("queryResult", {}).get("queryText", "")

    if not query_text:
        return jsonify(
            {
                "fulfillmentText": "Tôi không hiểu câu hỏi của bạn. Vui lòng hỏi lại về luật giao thông đường bộ."
            }
        )

    # Xử lý với AI
    result = chatbot.generate_response(query_text)

    return jsonify({"fulfillmentText": result["answer"]})


@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint cho direct chat"""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            logger.warning("Invalid request: missing message")
            return jsonify({"error": "Vui lòng gửi message trong request body"}), 400

        query = data.get("message", "").strip()
        logger.info(f"Received chat request: {query}")
        result = chatbot.generate_response(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/test", methods=["GET"])
def test():
    """Test endpoint"""
    test_queries = [
        "xe máy chạy 80km/h bị phạt bao nhiêu",
        "uống rượu lái ô tô",
        "không đội mũ bảo hiểm",
    ]

    results = {}
    for query in test_queries:
        results[query] = chatbot.generate_response(query)

    return jsonify(results)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "violations_loaded": len(chatbot.violations),
            "timestamp": str(datetime.now()),
        }
    )


@app.route("/violations", methods=["POST"])
def add_violation():
    """Add new violation"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Invalid request: no data provided")
            return jsonify({"error": "No data provided"}), 400

        logger.info("Adding new violation")
        violation_id = chatbot.add_violation(data)
        return jsonify({"id": violation_id, "message": "Violation added successfully"})
    except Exception as e:
        logger.error(f"Error adding violation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/violations/<int:violation_id>", methods=["PUT"])
def update_violation(violation_id):
    """Update violation"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    success = chatbot.update_violation(violation_id, data)
    if success:
        return jsonify({"message": "Violation updated successfully"})
    return jsonify({"error": "Violation not found"}), 404


@app.route("/violations/<int:violation_id>", methods=["DELETE"])
def delete_violation(violation_id):
    """Delete violation"""
    success = chatbot.delete_violation(violation_id)
    if success:
        return jsonify({"message": "Violation deleted successfully"})
    return jsonify({"error": "Violation not found"}), 404


@app.route("/legal-documents", methods=["POST"])
def add_legal_document():
    """Add new legal document"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    doc_id = chatbot.add_legal_document(data)
    return jsonify({"id": doc_id, "message": "Legal document added successfully"})


@app.route("/legal-documents/<int:doc_id>", methods=["PUT"])
def update_legal_document(doc_id):
    """Update legal document"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    success = chatbot.update_legal_document(doc_id, data)
    if success:
        return jsonify({"message": "Legal document updated successfully"})
    return jsonify({"error": "Legal document not found"}), 404


@app.route("/legal-documents/<int:doc_id>", methods=["DELETE"])
def delete_legal_document(doc_id):
    """Delete legal document"""
    success = chatbot.delete_legal_document(doc_id)
    if success:
        return jsonify({"message": "Legal document deleted successfully"})
    return jsonify({"error": "Legal document not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
