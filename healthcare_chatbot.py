# healthcare_chatbot.py
import os
import re
import sqlite3
from datetime import datetime
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logging_config import setup_logging

logger = setup_logging()


class HealthCareChatbot:
    def __init__(self):
        logger.info("Initializing HealthCareChatbot")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=1000, stop_words=None
        )
        self.db_path = "healthcare_database.db"
        self.setup_database()
        self.load_data()
        self.build_search_index()
        logger.info("HealthCareChatbot initialization completed")

    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    def setup_database(self):
        """Setup database tables and insert sample data"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS health_advice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    condition_name TEXT NOT NULL,
                    symptoms TEXT,
                    advice TEXT NOT NULL,
                    warning_level TEXT DEFAULT 'normal',
                    keywords TEXT,
                    age_group TEXT DEFAULT 'all',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nutrition_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    benefits TEXT,
                    nutritional_value TEXT,
                    recommended_for TEXT,
                    cautions TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS exercise_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    benefits TEXT,
                    duration_minutes INTEGER,
                    intensity_level TEXT,
                    suitable_for TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS emergency_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_name TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    immediate_action TEXT NOT NULL,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM health_advice")
            if cursor.fetchone()[0] == 0:
                self.insert_sample_data(cursor)

            conn.commit()
            conn.close()
            logger.info("Database setup completed successfully")

        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise

    def insert_sample_data(self, cursor):
        """Insert sample health data"""

        # Health advice data
        health_advice_data = [
            (
                "symptom",
                "Đau đầu thông thường",
                "đau đầu, nhức đầu, căng thẳng",
                "Nghỉ ngơi trong phòng tối, uống đủ nước, massage nhẹ thái dương. Nếu đau kéo dài hơn 2 ngày hoặc đau dữ dội, nên đến bệnh viện.",
                "normal",
                "đau đầu nhức đầu căng thẳng stress mệt mỏi",
                "all",
            ),
            (
                "symptom",
                "Sốt nhẹ",
                "sốt từ 37.5-38.5°C, mệt mỏi",
                "Nghỉ ngơi, uống nhiều nước, chườm mát, có thể dùng paracetamol theo hướng dẫn. Theo dõi nhiệt độ thường xuyên.",
                "caution",
                "sốt nhiệt độ cao mệt mỏi ốm",
                "all",
            ),
            (
                "symptom",
                "Ho khan",
                "ho không có đờm, khó chịu họng",
                "Uống nước ấm, mật ong, ngậm kẹo ho. Tránh khói thuốc, bụi bẩn. Nếu ho kéo dài hơn 2 tuần nên khám bác sĩ.",
                "normal",
                "ho khan họng khô viêm họng",
                "all",
            ),
            (
                "symptom",
                "Đau bụng nhẹ",
                "đau bụng không rõ nguyên nhân",
                "Nghỉ ngơi, uống nước ấm, chườm ấm bụng. Ăn nhẹ, tránh thức ăn cay nóng. Nếu đau dữ dội hoặc kèm sốt cao cần đi khám ngay.",
                "caution",
                "đau bụng đau dạ dày khó tiêu",
                "all",
            ),
            (
                "mental_health",
                "Stress căng thẳng",
                "lo âu, mệt mỏi tinh thần, khó ngủ",
                "Thực hành thở sâu, tập yoga, thiền. Duy trì lịch ngủ đều đặn, tập thể dục nhẹ. Nói chuyện với người thân hoặc chuyên gia tâm lý.",
                "normal",
                "stress căng thẳng lo âu trầm cảm tâm lý",
                "all",
            ),
            (
                "prevention",
                "Tăng cường miễn dịch",
                "phòng ngừa bệnh tật",
                "Ăn đủ chất dinh dưỡng, ngủ đủ giấc 7-8 tiếng/đêm, tập thể dục đều đặn, rửa tay thường xuyên, tiêm vacchin đầy đủ.",
                "normal",
                "miễn dịch sức khỏe phòng bệnh tăng cường",
                "all",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO health_advice (category, condition_name, symptoms, advice, warning_level, keywords, age_group)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            health_advice_data,
        )

        # Nutrition data
        nutrition_data = [
            (
                "Rau xanh",
                "vegetables",
                "Giàu vitamin A, C, K và chất xơ. Tốt cho tiêu hóa và thị lực.",
                "Ít calo, nhiều chất dinh dưỡng",
                "Người muốn giảm cân, tăng cường miễn dịch",
                "Người có vấn đề thận nên hạn chế rau có oxalate cao",
                "rau xanh rau cải rau muống vitamin",
            ),
            (
                "Trái cây",
                "fruits",
                "Cung cấp vitamin C, chất xơ và chất chống oxy hóa.",
                "Đường tự nhiên, vitamin, khoáng chất",
                "Mọi lứa tuổi",
                "Người tiểu đường nên ăn điều độ",
                "trái cây hoa quả vitamin C",
            ),
            (
                "Cá hồi",
                "protein",
                "Giàu omega-3, protein chất lượng cao, tốt cho tim mạch và não bộ.",
                "Protein 20g/100g, omega-3 cao",
                "Người cao tuổi, trẻ em phát triển",
                "Người dị ứng hải sản cần tránh",
                "cá hồi omega-3 protein tim mạch",
            ),
            (
                "Sữa",
                "dairy",
                "Cung cấp canxi, protein, vitamin D cho xương chắc khỏe.",
                "Canxi 120mg/100ml, protein 3.2g/100ml",
                "Trẻ em, người cao tuổi",
                "Người không dung nạp lactose nên chọn sữa không lactose",
                "sữa canxi xương protein",
            ),
            (
                "Yến mạch",
                "grains",
                "Giàu chất xơ beta-glucan, giúp giảm cholesterol và ổn định đường huyết.",
                "Chất xơ cao, carbohydrate phức",
                "Người tiểu đường, muốn giảm cân",
                "Không có tác dụng phụ đặc biệt",
                "yến mạch oats chất xơ cholesterol",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO nutrition_data (food_name, category, benefits, nutritional_value, recommended_for, cautions, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            nutrition_data,
        )

        # Exercise data
        exercise_data = [
            (
                "Đi bộ",
                "cardio",
                "Hoạt động thể chất nhẹ nhàng, phù hợp mọi lứa tuổi",
                "Tăng cường tim mạch, giảm căng thẳng, cải thiện giấc ngủ",
                30,
                "low",
                "Mọi lứa tuổi",
                "đi bộ walking cardio tim mạch",
            ),
            (
                "Chạy bộ",
                "cardio",
                "Hoạt động cardio cường độ trung bình đến cao",
                "Giảm cân hiệu quả, tăng cường sức bền, cải thiện tâm trạng",
                45,
                "medium",
                "Người trưởng thành khỏe mạnh",
                "chạy bộ running cardio giảm cân",
            ),
            (
                "Yoga",
                "flexibility",
                "Kết hợp giữa thể chất và tinh thần",
                "Tăng độ dẻo dai, giảm stress, cải thiện tư thế",
                60,
                "low",
                "Mọi lứa tuổi",
                "yoga thiền meditation linh hoạt",
            ),
            (
                "Bơi lội",
                "full_body",
                "Vận động toàn thân trong nước",
                "Tăng cường sức mạnh toàn thân, ít tác động lên khớp",
                45,
                "medium",
                "Mọi lứa tuổi, đặc biệt tốt cho người có vấn đề khớp",
                "bơi lội swimming toàn thân khớp",
            ),
            (
                "Tập tạ",
                "strength",
                "Tập luyện sức mạnh với trọng lượng",
                "Tăng khối lượng cơ, cải thiện mật độ xương, tăng trao đổi chất",
                60,
                "high",
                "Người trưởng thành, người muốn tăng cơ",
                "tập tạ gym sức mạnh cơ bắp",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO exercise_data (exercise_name, category, description, benefits, duration_minutes, intensity_level, suitable_for, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            exercise_data,
        )

        # Emergency conditions
        emergency_data = [
            (
                "Đau ngực dữ dội",
                "đau ngực như bị ép, khó thở, toát mồ hôi",
                "Gọi cấp cứu 115 NGAY LẬP TỨC. Ngồi nghiêng về phía trước, nới lỏng quần áo. KHÔNG tự lái xe đến bệnh viện.",
                "đau ngực tim mạch cấp cứu khó thở",
            ),
            (
                "Sốt cao trên 39°C",
                "sốt 39-40°C, lú lẫn, co giật",
                "Gọi cấp cứu hoặc đến bệnh viện ngay. Chườm mát, cởi bớt quần áo, uống nước nếu tỉnh táo.",
                "sốt cao nhiệt độ co giật lú lẫn",
            ),
            (
                "Khó thở nghiêm trọng",
                "thở gấp, không nói được câu hoàn chỉnh, tím môi",
                "Gọi 115 ngay. Giúp bệnh nhân ngồi thẳng, nới lỏng quần áo quanh cổ và ngực.",
                "khó thở thở gấp tím tái cấp cứu",
            ),
            (
                "Đột quỵ",
                "méo miệng, tê liệt một bên, nói khó, đau đầu dữ dội đột ngột",
                "Gọi 115 NGAY. Để bệnh nhân nằm nghiêng, không cho ăn uống gì. Ghi nhận thời gian xuất hiện triệu chứng.",
                "đột quỵ tai biến méo miệng tê liệt",
            ),
            (
                "Chấn thương đầu nghiêm trọng",
                "bất tỉnh, nôn mửa, lú lẫn sau va đập đầu",
                "Gọi 115. KHÔNG di chuyển bệnh nhân. Giữ đầu và cổ thẳng, quan sát hô hấp.",
                "chấn thương đầu bất tỉnh não nôn mửa",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO emergency_conditions (condition_name, symptoms, immediate_action, keywords)
            VALUES (?, ?, ?, ?)
        """,
            emergency_data,
        )

        logger.info("Sample health data inserted successfully")

    def load_data(self):
        """Load data from database"""
        try:
            logger.info("Loading health data from database")
            conn = self.get_db_connection()

            self.health_advice = pd.read_sql_query("SELECT * FROM health_advice", conn)
            self.nutrition_data = pd.read_sql_query(
                "SELECT * FROM nutrition_data", conn
            )
            self.exercise_data = pd.read_sql_query("SELECT * FROM exercise_data", conn)
            self.emergency_conditions = pd.read_sql_query(
                "SELECT * FROM emergency_conditions", conn
            )

            conn.close()

            logger.info(
                f"Loaded {len(self.health_advice)} health advice, "
                f"{len(self.nutrition_data)} nutrition data, "
                f"{len(self.exercise_data)} exercise data, "
                f"{len(self.emergency_conditions)} emergency conditions"
            )

            # Create corpus for search
            self.corpus = []

            # Add health advice to corpus
            for _, advice in self.health_advice.iterrows():
                text = f"{advice['condition_name']} {advice['symptoms']} {advice['keywords']} {advice['category']}"
                self.corpus.append(text.lower())

            # Add nutrition data to corpus
            for _, nutrition in self.nutrition_data.iterrows():
                text = f"{nutrition['food_name']} {nutrition['benefits']} {nutrition['keywords']} {nutrition['category']}"
                self.corpus.append(text.lower())

            # Add exercise data to corpus
            for _, exercise in self.exercise_data.iterrows():
                text = f"{exercise['exercise_name']} {exercise['benefits']} {exercise['keywords']} {exercise['category']}"
                self.corpus.append(text.lower())

        except Exception as e:
            logger.error(f"Error loading health data: {str(e)}")
            raise

    def build_search_index(self):
        """Build vector search index"""
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            logger.info("Health search index created successfully")
        else:
            logger.warning("No data available to create search index")

    def preprocess_text(self, text):
        """Preprocess Vietnamese health-related text"""
        text = text.lower()

        # Health-specific replacements
        replacements = {
            "đau đầu": "nhức đầu",
            "nhức đầu": "đau đầu",
            "cảm cúm": "cảm lạnh",
            "cảm lạnh": "cảm cúm",
            "ho có đờm": "ho có đờm balgam",
            "ho khan": "ho khô",
            "sốt cao": "nhiệt độ cao",
            "đau bụng": "đau dạ dày",
            "mệt mỏi": "mệt mỏi uể oải",
            "uể oải": "mệt mỏi uể oải",
            "căng thẳng": "stress",
            "stress": "căng thẳng stress",
            "lo âu": "lo lắng",
            "lo lắng": "lo âu lo lắng",
            "khó ngủ": "mất ngủ",
            "mất ngủ": "khó ngủ mất ngủ",
            "tập thể dục": "vận động",
            "thể thao": "vận động",
            "ăn uống": "dinh dưỡng",
            "chế độ ăn": "dinh dưỡng",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def extract_health_entities(self, text):
        """Extract health-related entities from query"""
        text = self.preprocess_text(text)

        entities = {
            "symptoms": [],
            "body_parts": [],
            "health_topics": [],
            "urgency_level": "normal",
        }

        # Detect symptoms
        symptom_keywords = [
            "đau đầu",
            "nhức đầu",
            "sốt",
            "ho",
            "đau bụng",
            "buồn nôn",
            "chóng mặt",
            "mệt mỏi",
            "khó ngủ",
            "căng thẳng",
            "lo âu",
            "uể oải",
            "mất ngủ",
            "stress",
            "lo lắng",
        ]

        for symptom in symptom_keywords:
            if symptom in text:
                entities["symptoms"].append(symptom)

        # Detect body parts
        body_parts = [
            "đầu",
            "cổ",
            "vai",
            "lưng",
            "bụng",
            "chân",
            "tay",
            "mắt",
            "tai",
            "mũi",
            "họng",
            "ngực",
            "tim",
            "phổi",
        ]

        for part in body_parts:
            if part in text:
                entities["body_parts"].append(part)

        # Detect health topics
        if any(word in text for word in ["ăn", "thức ăn", "dinh dưỡng", "vitamin"]):
            entities["health_topics"].append("nutrition")

        if any(word in text for word in ["tập", "thể dục", "vận động", "gym"]):
            entities["health_topics"].append("exercise")

        # Detect urgency
        emergency_keywords = [
            "cấp cứu",
            "khẩn cấp",
            "nguy hiểm",
            "dữ dội",
            "nghiêm trọng",
            "bất tỉnh",
            "choáng váng",
            "khó thở",
        ]

        if any(keyword in text for keyword in emergency_keywords):
            entities["urgency_level"] = "emergency"

        return entities

    def search_health_info(self, query, top_k=3):
        """Search for relevant health information"""
        if not self.corpus:
            return []

        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        entities = self.extract_health_entities(query)

        total_records = (
            len(self.health_advice) + len(self.nutrition_data) + len(self.exercise_data)
        )

        for idx in top_indices:
            if similarities[idx] > 0.05:
                # Determine which dataset this index belongs to
                if idx < len(self.health_advice):
                    # Health advice
                    record = self.health_advice.iloc[idx].to_dict()
                    record["data_type"] = "health_advice"
                    record["urgency_level"] = record.get("warning_level", "normal")
                elif idx < len(self.health_advice) + len(self.nutrition_data):
                    # Nutrition data
                    nutrition_idx = idx - len(self.health_advice)
                    record = self.nutrition_data.iloc[nutrition_idx].to_dict()
                    record["data_type"] = "nutrition"
                    record["urgency_level"] = "normal"
                else:
                    # Exercise data
                    exercise_idx = (
                        idx - len(self.health_advice) - len(self.nutrition_data)
                    )
                    record = self.exercise_data.iloc[exercise_idx].to_dict()
                    record["data_type"] = "exercise"
                    record["urgency_level"] = "normal"

                record["confidence"] = float(similarities[idx])

                # Bonus points for matching symptoms
                if entities["symptoms"]:
                    for symptom in entities["symptoms"]:
                        if (
                            "symptoms" in record
                            and record["symptoms"]
                            and symptom in record["symptoms"]
                        ):
                            record["confidence"] += 0.3

                results.append(record)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:3]

    def check_emergency(self, query):
        """Check if query indicates emergency condition"""
        processed_query = self.preprocess_text(query)

        # Check for emergency keywords first
        emergency_keywords = [
            "cấp cứu",
            "khẩn cấp",
            "nguy hiểm",
            "dữ dội",
            "nghiêm trọng",
            "bất tỉnh",
            "choáng váng",
            "khó thở",
        ]

        is_emergency = any(keyword in processed_query for keyword in emergency_keywords)

        if not is_emergency:
            return None

        for _, condition in self.emergency_conditions.iterrows():
            keywords = condition["keywords"].split()
            if any(keyword in processed_query for keyword in keywords):
                return condition.to_dict()

        return None

    def generate_health_response(self, query):
        """Generate health consultation response"""
        try:
            logger.info(f"Processing health query: {query}")

            if not query.strip():
                logger.warning("Empty query received")
                return {
                    "answer": "Vui lòng nhập câu hỏi sức khỏe của bạn.",
                    "confidence": 0.0,
                }

            # Check for emergency first
            emergency = self.check_emergency(query)
            if emergency:
                response = f"🚨 **CẢNH BÁO KHẨN CẤP - {emergency['condition_name'].upper()}**\n\n"
                response += f"**Triệu chứng:** {emergency['symptoms']}\n\n"
                response += (
                    f"**HÀNH ĐỘNG NGAY LẬP TỨC:**\n{emergency['immediate_action']}\n\n"
                )
                response += (
                    "⚠️ **KHÔNG trì hoãn, hãy tìm kiếm sự giúp đỡ y tế ngay lập tức!**"
                )

                return {
                    "answer": response,
                    "confidence": 1.0,
                    "urgency": "emergency",
                    "emergency_condition": emergency["condition_name"],
                }

            # Normal search
            results = self.search_health_info(query)
            entities = self.extract_health_entities(query)

            if not results:
                logger.info("No health information found for query")
                suggestion = "Bạn có thể hỏi về:\n"
                suggestion += "- Các triệu chứng thông thường (đau đầu, sốt, ho...)\n"
                suggestion += "- Tư vấn dinh dưỡng (thực phẩm tốt cho sức khỏe)\n"
                suggestion += "- Hoạt động thể chất (bài tập phù hợp)\n"
                suggestion += "- Cách xử lý stress và chăm sóc tinh thần"

                return {
                    "answer": f'Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi "{query}".\n\n{suggestion}',
                    "confidence": 0.0,
                }

            # Build response
            response = f'**Tư vấn sức khỏe cho câu hỏi: "{query}"**\n\n'

            for i, result in enumerate(results, 1):
                if result["data_type"] == "health_advice":
                    response += f"**{i}. {result['condition_name']}**\n"
                    if result["symptoms"]:
                        response += f"🔸 **Triệu chứng:** {result['symptoms']}\n"
                    response += f"💡 **Lời khuyên:** {result['advice']}\n"
                    if result.get("warning_level") == "caution":
                        response += (
                            "⚠️ **Lưu ý:** Cần theo dõi và có thể cần tư vấn bác sĩ\n"
                        )
                    elif result.get("warning_level") == "emergency":
                        response += "🚨 **Cảnh báo:** Cần đến cơ sở y tế ngay lập tức\n"

                elif result["data_type"] == "nutrition":
                    response += f"**{i}. {result['food_name']} - Dinh Dưỡng**\n"
                    response += f"🥗 **Lợi ích:** {result['benefits']}\n"
                    response += (
                        f"📊 **Giá trị dinh dưỡng:** {result['nutritional_value']}\n"
                    )
                    response += f"👥 **Phù hợp cho:** {result['recommended_for']}\n"
                    if result.get("cautions"):
                        response += f"⚠️ **Lưu ý:** {result['cautions']}\n"

                elif result["data_type"] == "exercise":
                    response += f"**{i}. {result['exercise_name']} - Thể Chất**\n"
                    response += f"🏃 **Mô tả:** {result['description']}\n"
                    response += f"💪 **Lợi ích:** {result['benefits']}\n"
                    response += f"⏱️ **Thời gian:** {result['duration_minutes']} phút\n"
                    response += f"📈 **Cường độ:** {result['intensity_level']}\n"
                    response += f"👥 **Phù hợp cho:** {result['suitable_for']}\n"

                response += "\n"

            response += "---\n"
            response += "**⚠️ QUAN TRỌNG:** Thông tin này chỉ mang tính chất tham khảo và không thay thế cho việc khám bác sĩ. "
            response += "Nếu có triệu chứng nghiêm trọng hoặc kéo dài, hãy tìm kiếm tư vấn y tế chuyên nghiệp."

            return {
                "answer": response,
                "confidence": results[0]["confidence"] if results else 0.0,
                "results_found": len(results),
                "entities": entities,
                "urgency": entities.get("urgency_level", "normal"),
            }

        except Exception as e:
            logger.error(f"Error generating health response: {str(e)}")
            return {
                "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi sức khỏe của bạn.",
                "confidence": 0.0,
                "urgency": "normal",
            }

    @lru_cache(maxsize=100)
    def cached_health_search(self, query):
        """Cached search for better performance"""
        return self.search_health_info(query)

    # Data management methods
    def add_health_advice(self, advice_data):
        """Add new health advice"""
        try:
            logger.info(f"Adding new health advice: {advice_data['condition_name']}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO health_advice (category, condition_name, symptoms, advice, warning_level, keywords, age_group)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    advice_data["category"],
                    advice_data["condition_name"],
                    advice_data.get("symptoms", ""),
                    advice_data["advice"],
                    advice_data.get("warning_level", "normal"),
                    advice_data["keywords"],
                    advice_data.get("age_group", "all"),
                ),
            )

            advice_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully added health advice with ID: {advice_id}")
            return advice_id

        except Exception as e:
            logger.error(f"Error adding health advice: {str(e)}")
            raise

    def add_nutrition_data(self, nutrition_data):
        """Add new nutrition data"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO nutrition_data (food_name, category, benefits, nutritional_value, recommended_for, cautions)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    nutrition_data["food_name"],
                    nutrition_data["category"],
                    nutrition_data["benefits"],
                    nutrition_data["nutritional_value"],
                    nutrition_data["recommended_for"],
                    nutrition_data.get("cautions", ""),
                ),
            )

            nutrition_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully added nutrition data with ID: {nutrition_id}")
            return nutrition_id

        except Exception as e:
            logger.error(f"Error adding nutrition data: {str(e)}")
            raise

    def add_exercise_data(self, exercise_data):
        """Add new exercise data"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO exercise_data (exercise_name, category, description, benefits, duration_minutes, intensity_level, suitable_for)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    exercise_data["exercise_name"],
                    exercise_data["category"],
                    exercise_data["description"],
                    exercise_data["benefits"],
                    exercise_data.get("duration_minutes", 30),
                    exercise_data.get("intensity_level", "medium"),
                    exercise_data["suitable_for"],
                ),
            )

            exercise_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully added exercise data with ID: {exercise_id}")
            return exercise_id

        except Exception as e:
            logger.error(f"Error adding exercise data: {str(e)}")
            raise

    def add_emergency_condition(self, condition_data):
        """Add new emergency condition"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO emergency_conditions (condition_name, symptoms, immediate_action, keywords)
                VALUES (?, ?, ?, ?)
            """,
                (
                    condition_data["condition_name"],
                    condition_data["symptoms"],
                    condition_data["immediate_action"],
                    condition_data.get("keywords", ""),
                ),
            )

            condition_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(
                f"Successfully added emergency condition with ID: {condition_id}"
            )
            return condition_id

        except Exception as e:
            logger.error(f"Error adding emergency condition: {str(e)}")
            raise

    def update_health_advice(self, advice_id, updated_data):
        """Update existing health advice"""
        try:
            logger.info(f"Updating health advice ID: {advice_id}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE health_advice
                SET category = ?, condition_name = ?, symptoms = ?, advice = ?, warning_level = ?, keywords = ?, age_group = ?
                WHERE id = ?
            """,
                (
                    updated_data["category"],
                    updated_data["condition_name"],
                    updated_data.get("symptoms", ""),
                    updated_data["advice"],
                    updated_data.get("warning_level", "normal"),
                    updated_data["keywords"],
                    updated_data.get("age_group", "all"),
                    advice_id,
                ),
            )

            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully updated health advice ID: {advice_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating health advice ID {advice_id}: {str(e)}")
            raise

    def update_nutrition_data(self, nutrition_id, updated_data):
        """Update existing nutrition data"""
        try:
            logger.info(f"Updating nutrition data ID: {nutrition_id}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE nutrition_data
                SET food_name = ?, category = ?, benefits = ?, nutritional_value = ?, recommended_for = ?, cautions = ?
                WHERE id = ?
            """,
                (
                    updated_data["food_name"],
                    updated_data["category"],
                    updated_data["benefits"],
                    updated_data["nutritional_value"],
                    updated_data["recommended_for"],
                    updated_data.get("cautions", ""),
                    nutrition_id,
                ),
            )

            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully updated nutrition data ID: {nutrition_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating nutrition data ID {nutrition_id}: {str(e)}")
            raise

    def update_exercise_data(self, exercise_id, updated_data):
        """Update existing exercise data"""
        try:
            logger.info(f"Updating exercise data ID: {exercise_id}")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE exercise_data
                SET exercise_name = ?, category = ?, description = ?, benefits = ?, duration_minutes = ?, intensity_level = ?, suitable_for = ?
                WHERE id = ?
            """,
                (
                    updated_data["exercise_name"],
                    updated_data["category"],
                    updated_data["description"],
                    updated_data["benefits"],
                    updated_data.get("duration_minutes", 30),
                    updated_data.get("intensity_level", "medium"),
                    updated_data["suitable_for"],
                    exercise_id,
                ),
            )

            conn.commit()
            conn.close()
            self.load_data()
            self.build_search_index()

            logger.info(f"Successfully updated exercise data ID: {exercise_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating exercise data ID {exercise_id}: {str(e)}")
            raise
