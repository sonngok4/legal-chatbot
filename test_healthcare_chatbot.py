import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from healthcare_chatbot import HealthCareChatbot


class TestHealthCareChatbot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary database for testing
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db_path = self.test_db.name
        self.test_db.close()

        # Create test database schema and data
        self.setup_test_database()

        # Mock the database path and initialization
        with patch.object(HealthCareChatbot, "__init__", return_value=None):
            self.chatbot = HealthCareChatbot()
            self.chatbot.db_path = self.test_db_path
            self.chatbot.vectorizer = MagicMock()
            self.chatbot.health_advice = pd.DataFrame()
            self.chatbot.nutrition_data = pd.DataFrame()
            self.chatbot.exercise_data = pd.DataFrame()
            self.chatbot.emergency_conditions = pd.DataFrame()
            self.chatbot.corpus = []
            self.chatbot.tfidf_matrix = None

    def tearDown(self):
        """Clean up after each test method."""
        # Ensure all connections are closed before deleting the file
        try:
            if hasattr(self, "chatbot") and hasattr(self.chatbot, "db_path"):
                conn = sqlite3.connect(self.chatbot.db_path)
                conn.close()
        except Exception:
            pass
        if os.path.exists(self.test_db_path):
            try:
                os.unlink(self.test_db_path)
            except Exception:
                pass

    def mock_init(self, chatbot_self):
        """Mock initialization to use test database."""
        chatbot_self.db_path = self.test_db_path
        chatbot_self.vectorizer = MagicMock()
        chatbot_self.health_advice = pd.DataFrame()
        chatbot_self.nutrition_data = pd.DataFrame()
        chatbot_self.exercise_data = pd.DataFrame()
        chatbot_self.emergency_conditions = pd.DataFrame()
        chatbot_self.corpus = []
        chatbot_self.tfidf_matrix = None

    def setup_test_database(self):
        """Create test database with sample health data."""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create tables with IF NOT EXISTS
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS health_advice (
                id INTEGER PRIMARY KEY,
                category TEXT,
                condition_name TEXT,
                symptoms TEXT,
                advice TEXT,
                warning_level TEXT,
                keywords TEXT,
                age_group TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nutrition_data (
                id INTEGER PRIMARY KEY,
                food_name TEXT,
                category TEXT,
                benefits TEXT,
                nutritional_value TEXT,
                recommended_for TEXT,
                cautions TEXT,
                keywords TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exercise_data (
                id INTEGER PRIMARY KEY,
                exercise_name TEXT,
                category TEXT,
                description TEXT,
                benefits TEXT,
                duration_minutes INTEGER,
                intensity_level TEXT,
                suitable_for TEXT,
                keywords TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emergency_conditions (
                id INTEGER PRIMARY KEY,
                condition_name TEXT,
                symptoms TEXT,
                immediate_action TEXT,
                keywords TEXT
            )
        """
        )

        # Clear all tables before inserting new data
        cursor.execute("DELETE FROM health_advice")
        cursor.execute("DELETE FROM nutrition_data")
        cursor.execute("DELETE FROM exercise_data")
        cursor.execute("DELETE FROM emergency_conditions")

        # Insert sample data
        sample_health_advice = [
            (
                1,
                "symptom",
                "Đau đầu",
                "đau đầu, nhức đầu",
                "Nghỉ ngơi, uống nước",
                "normal",
                "đau đầu nhức",
                "all",
            ),
            (
                2,
                "symptom",
                "Sốt nhẹ",
                "sốt 37-38°C",
                "Uống nhiều nước, nghỉ ngơi",
                "caution",
                "sốt nhiệt độ",
                "all",
            ),
        ]

        sample_nutrition = [
            (
                1,
                "Rau xanh",
                "vegetables",
                "Giàu vitamin",
                "Vitamin A, C",
                "Mọi người",
                "Không có",
                "rau xanh vitamin",
            ),
            (
                2,
                "Cá hồi",
                "protein",
                "Omega-3 cao",
                "Protein, omega-3",
                "Người lớn",
                "Dị ứng hải sản",
                "cá omega protein",
            ),
        ]

        sample_exercise = [
            (
                1,
                "Đi bộ",
                "cardio",
                "Vận động nhẹ",
                "Tốt cho tim",
                30,
                "low",
                "Mọi người",
                "đi bộ cardio",
            ),
            (
                2,
                "Yoga",
                "flexibility",
                "Tăng linh hoạt",
                "Giảm stress",
                60,
                "low",
                "Mọi người",
                "yoga thiền",
            ),
        ]

        sample_emergency = [
            (1, "Đau ngực", "đau ngực dữ dội", "Gọi cấp cứu 115", "đau ngực tim"),
            (
                2,
                "Khó thở",
                "thở gấp, tím tái",
                "Gọi 115, ngồi thẳng",
                "khó thở cấp cứu",
            ),
        ]

        cursor.executemany(
            "INSERT INTO health_advice VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sample_health_advice,
        )
        cursor.executemany(
            "INSERT INTO nutrition_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sample_nutrition,
        )
        cursor.executemany(
            """
            INSERT INTO exercise_data (id, exercise_name, category, description, benefits, duration_minutes, intensity_level, suitable_for, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_exercise,
        )
        cursor.executemany(
            "INSERT INTO emergency_conditions VALUES (?, ?, ?, ?, ?)", sample_emergency
        )

        conn.commit()
        conn.close()

    def test_get_db_connection(self):
        """Test database connection."""
        conn = self.chatbot.get_db_connection()
        self.assertIsNotNone(conn)
        conn.close()

    def test_setup_database(self):
        """Test database setup."""
        # This should not raise any exceptions
        self.chatbot.setup_database()

    def test_load_data(self):
        """Test data loading from database."""
        self.setup_test_database()
        self.chatbot.load_data()

        self.assertIsInstance(self.chatbot.health_advice, pd.DataFrame)
        self.assertIsInstance(self.chatbot.nutrition_data, pd.DataFrame)
        self.assertIsInstance(self.chatbot.exercise_data, pd.DataFrame)
        self.assertIsInstance(self.chatbot.emergency_conditions, pd.DataFrame)

    def test_preprocess_text(self):
        """Test Vietnamese health text preprocessing."""
        test_cases = [
            ("Đau đầu", "nhức đầu"),
            ("Cảm cúm", "cảm lạnh"),
            ("Tập thể dục", "vận động"),
            ("Ăn uống", "dinh dưỡng"),
        ]

        for input_text, expected in test_cases:
            result = self.chatbot.preprocess_text(input_text)
            # Check if expected terms are in the result
            self.assertTrue(
                any(term in result for term in [expected, input_text.lower()])
            )

    def test_extract_health_entities(self):
        """Test health entity extraction."""
        test_cases = [
            {"query": "đau đầu mệt mỏi", "expected_symptoms": ["đau đầu", "mệt mỏi"]},
            {"query": "tập thể dục yoga", "expected_topics": ["exercise"]},
            {"query": "ăn rau xanh vitamin", "expected_topics": ["nutrition"]},
            {"query": "cấp cứu khẩn cấp", "expected_urgency": "emergency"},
        ]

        for case in test_cases:
            entities = self.chatbot.extract_health_entities(case["query"])

            if "expected_symptoms" in case:
                for symptom in case["expected_symptoms"]:
                    self.assertIn(symptom, entities.get("symptoms", []))

            if "expected_topics" in case:
                for topic in case["expected_topics"]:
                    self.assertIn(topic, entities.get("health_topics", []))

            if "expected_urgency" in case:
                self.assertEqual(
                    entities.get("urgency_level"), case["expected_urgency"]
                )

    @patch("healthcare_chatbot.TfidfVectorizer")
    @patch("healthcare_chatbot.cosine_similarity")
    def test_search_health_info(self, mock_cosine, mock_vectorizer):
        """Test health information search."""
        # Setup mocks
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vectorizer_instance.transform.return_value = MagicMock()
        mock_cosine.return_value = np.array([[0.8, 0.6, 0.3, 0.2, 0.1, 0.05]])

        # Setup test data
        self.setup_test_database()
        self.chatbot.load_data()
        self.chatbot.vectorizer = mock_vectorizer_instance
        self.chatbot.tfidf_matrix = MagicMock()

        # Test search
        results = self.chatbot.search_health_info("đau đầu")
        self.assertIsInstance(results, list)

    def test_check_emergency(self):
        """Test emergency condition detection."""
        self.setup_test_database()
        self.chatbot.load_data()

        # Test emergency detection
        emergency = self.chatbot.check_emergency("đau ngực dữ dội")
        self.assertIsNotNone(emergency)
        self.assertEqual(emergency["condition_name"], "Đau ngực")

        # Test non-emergency
        non_emergency = self.chatbot.check_emergency("đau đầu nhẹ")
        self.assertIsNone(non_emergency)

    def test_generate_health_response_empty_query(self):
        """Test response generation with empty query."""
        response = self.chatbot.generate_health_response("")

        self.assertIn("answer", response)
        self.assertEqual(response["confidence"], 0.0)
        self.assertIn("Vui lòng nhập câu hỏi", response["answer"])

    def test_generate_health_response_emergency(self):
        """Test response generation for emergency conditions."""
        self.setup_test_database()
        self.chatbot.load_data()

        response = self.chatbot.generate_health_response("đau ngực dữ dội")

        self.assertIn("answer", response)
        self.assertEqual(response["confidence"], 1.0)
        self.assertEqual(response["urgency"], "emergency")
        self.assertIn("CẢNH BÁO KHẨN CẤP", response["answer"])

    def test_generate_health_response_no_results(self):
        """Test response generation when no results found."""
        with patch.object(self.chatbot, "check_emergency", return_value=None):
            with patch.object(self.chatbot, "search_health_info", return_value=[]):
                response = self.chatbot.generate_health_response("unknown condition")

                self.assertIn("answer", response)
                self.assertEqual(response["confidence"], 0.0)
                self.assertIn("không tìm thấy thông tin", response["answer"])

    def test_generate_health_response_with_results(self):
        """Test response generation with search results."""
        mock_results = [
            {
                "data_type": "health_advice",
                "condition_name": "Test condition",
                "symptoms": "Test symptoms",
                "advice": "Test advice",
                "warning_level": "normal",
                "urgency_level": "normal",
                "confidence": 0.8,
            }
        ]

        with patch.object(self.chatbot, "check_emergency", return_value=None):
            with patch.object(
                self.chatbot, "search_health_info", return_value=mock_results
            ):
                with patch.object(
                    self.chatbot, "extract_health_entities", return_value={}
                ):
                    response = self.chatbot.generate_health_response("test query")

                    self.assertIn("answer", response)
                    self.assertGreater(response["confidence"], 0)
                    self.assertIn("Test condition", response["answer"])

    def test_cached_health_search(self):
        """Test cached search functionality."""
        with patch.object(
            self.chatbot, "search_health_info", return_value=[]
        ) as mock_search:
            # First call
            result1 = self.chatbot.cached_health_search("test query")
            # Second call should use cache
            result2 = self.chatbot.cached_health_search("test query")

            # Should only call search_health_info once due to caching
            mock_search.assert_called_once()
            self.assertEqual(result1, result2)

    def test_add_health_advice(self):
        """Test adding new health advice."""
        self.setup_test_database()

        advice_data = {
            "category": "test",
            "condition_name": "Test condition",
            "advice": "Test advice",
            "keywords": "test keywords",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                advice_id = self.chatbot.add_health_advice(advice_data)
                self.assertIsInstance(advice_id, int)
                self.assertGreater(advice_id, 0)

    def test_add_nutrition_data(self):
        """Test adding new nutrition data."""
        self.setup_test_database()

        nutrition_data = {
            "food_name": "Test food",
            "category": "test",
            "benefits": "Test benefits",
            "nutritional_value": "Test value",
            "recommended_for": "Test group",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                nutrition_id = self.chatbot.add_nutrition_data(nutrition_data)
                self.assertIsInstance(nutrition_id, int)
                self.assertGreater(nutrition_id, 0)

    def test_add_exercise_data(self):
        """Test adding new exercise data."""
        self.setup_test_database()

        exercise_data = {
            "exercise_name": "Test exercise",
            "category": "test",
            "description": "Test description",
            "benefits": "Test benefits",
            "suitable_for": "Test group",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                exercise_id = self.chatbot.add_exercise_data(exercise_data)
                self.assertIsInstance(exercise_id, int)
                self.assertGreater(exercise_id, 0)

    def test_add_emergency_condition(self):
        """Test adding new emergency condition."""
        self.setup_test_database()

        condition_data = {
            "condition_name": "Test emergency",
            "symptoms": "Test symptoms",
            "immediate_action": "Test action",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                condition_id = self.chatbot.add_emergency_condition(condition_data)
                self.assertIsInstance(condition_id, int)
                self.assertGreater(condition_id, 0)

    def test_update_health_advice(self):
        """Test updating health advice."""
        self.setup_test_database()

        update_data = {
            "category": "updated",
            "condition_name": "Updated condition",
            "advice": "Updated advice",
            "keywords": "updated keywords",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                success = self.chatbot.update_health_advice(1, update_data)
                self.assertTrue(success)

    def test_update_nutrition_data(self):
        """Test updating nutrition data."""
        self.setup_test_database()

        update_data = {
            "food_name": "Updated food",
            "category": "updated",
            "benefits": "Updated benefits",
            "nutritional_value": "Updated value",
            "recommended_for": "Updated group",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                success = self.chatbot.update_nutrition_data(1, update_data)
                self.assertTrue(success)

    def test_update_exercise_data(self):
        """Test updating exercise data."""
        self.setup_test_database()

        update_data = {
            "exercise_name": "Updated exercise",
            "category": "updated",
            "description": "Updated description",
            "benefits": "Updated benefits",
            "suitable_for": "Updated group",
        }

        with patch.object(self.chatbot, "load_data"):
            with patch.object(self.chatbot, "build_search_index"):
                success = self.chatbot.update_exercise_data(1, update_data)
                self.assertTrue(success)

    def test_nutrition_response_format(self):
        """Test nutrition data response formatting."""
        mock_results = [
            {
                "data_type": "nutrition",
                "food_name": "Test Food",
                "benefits": "Test benefits",
                "nutritional_value": "Test nutrition",
                "recommended_for": "Test group",
                "cautions": "Test cautions",
                "urgency_level": "normal",
                "confidence": 0.8,
            }
        ]

        with patch.object(self.chatbot, "check_emergency", return_value=None):
            with patch.object(
                self.chatbot, "search_health_info", return_value=mock_results
            ):
                with patch.object(
                    self.chatbot, "extract_health_entities", return_value={}
                ):
                    response = self.chatbot.generate_health_response("test nutrition")

                    self.assertIn("Test Food", response["answer"])
                    self.assertIn("Lợi ích:", response["answer"])
                    self.assertIn("Giá trị dinh dưỡng:", response["answer"])

    def test_exercise_response_format(self):
        """Test exercise data response formatting."""
        mock_results = [
            {
                "data_type": "exercise",
                "exercise_name": "Test Exercise",
                "description": "Test description",
                "benefits": "Test benefits",
                "duration_minutes": 30,
                "intensity_level": "medium",
                "suitable_for": "Test group",
                "urgency_level": "normal",
                "confidence": 0.8,
            }
        ]

        with patch.object(self.chatbot, "check_emergency", return_value=None):
            with patch.object(
                self.chatbot, "search_health_info", return_value=mock_results
            ):
                with patch.object(
                    self.chatbot, "extract_health_entities", return_value={}
                ):
                    response = self.chatbot.generate_health_response("test exercise")

                    self.assertIn("Test Exercise", response["answer"])
                    self.assertIn("Mô tả:", response["answer"])
                    self.assertIn("Lợi ích:", response["answer"])
                    self.assertIn("30 phút", response["answer"])

    def test_warning_levels(self):
        """Test different warning levels in responses."""
        mock_results = [
            {
                "data_type": "health_advice",
                "condition_name": "Test condition",
                "symptoms": "Test symptoms",
                "advice": "Test advice",
                "warning_level": "caution",
                "urgency_level": "normal",
                "confidence": 0.8,
            }
        ]

        with patch.object(self.chatbot, "check_emergency", return_value=None):
            with patch.object(
                self.chatbot, "search_health_info", return_value=mock_results
            ):
                with patch.object(
                    self.chatbot, "extract_health_entities", return_value={}
                ):
                    response = self.chatbot.generate_health_response("test condition")

                    self.assertIn("Cần theo dõi", response["answer"])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
