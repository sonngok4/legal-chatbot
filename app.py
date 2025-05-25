from flask import Flask, request, jsonify
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép CORS để Flutter có thể gọi API


class LegalChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=1000, stop_words=None  # Unigram và bigram
        )
        self.load_data()
        self.build_search_index()

    def load_data(self):
        """Load dữ liệu từ JSON file"""
        try:
            with open("violations.json", "r", encoding="utf-8") as f:
                self.violations = json.load(f)
            print(f"Đã load {len(self.violations)} vi phạm")
        except FileNotFoundError:
            print("Không tìm thấy file violations.json")
            self.violations = []

        # Tạo corpus để tìm kiếm
        self.corpus = []
        for violation in self.violations:
            text = f"{violation['description']} {violation['keywords']} {violation['vehicle_type']} {violation['violation_type']}"
            self.corpus.append(text.lower())

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
                violation = self.violations[idx].copy()
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
        if not query.strip():
            return {"answer": "Vui lòng nhập câu hỏi của bạn.", "confidence": 0.0}

        violations = self.search_violations(query)
        entities = self.extract_entities(query)

        if not violations:
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
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Vui lòng gửi message trong request body"}), 400

    query = data.get("message", "").strip()
    result = chatbot.generate_response(query)

    return jsonify(result)


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
