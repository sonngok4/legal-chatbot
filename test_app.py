import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from app import app
from healthcare_chatbot import HealthCareChatbot
from traffic_law_chatbot import TrafficLawChatbot


class TestNewsAPIEndpoints(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = app.test_client()
        self.app.testing = True

        # Create a temporary database for testing
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db_path = self.test_db.name
        self.test_db.close()

        # Create test database schema and data
        self.setup_test_database()

        # Create mock initialization function that captures test_db_path
        def create_mock_init(test_db_path):
            def mock_init(self):
                self.db_path = test_db_path
                self.vectorizer = MagicMock()
                self.violations = pd.DataFrame()
                self.legal_documents = pd.DataFrame()
                self.corpus = []
                self.tfidf_matrix = None

            return mock_init

        # Mock the database path and initialization for both chatbots
        with patch.object(
            TrafficLawChatbot, "__init__", create_mock_init(self.test_db_path)
        ), patch.object(
            HealthCareChatbot, "__init__", create_mock_init(self.test_db_path)
        ):
            self.traffic_bot = TrafficLawChatbot()
            self.health_bot = HealthCareChatbot()

        # Patch the global chatbot instances in app.py
        self.traffic_bot_patcher = patch("app.traffic_bot", self.traffic_bot)
        self.health_bot_patcher = patch("app.health_bot", self.health_bot)
        self.traffic_bot_patcher.start()
        self.health_bot_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop the patches
        self.traffic_bot_patcher.stop()
        self.health_bot_patcher.stop()

        # Remove the temporary database file
        if os.path.exists(self.test_db_path):
            os.unlink(self.test_db_path)

    def setup_test_database(self):
        """Create test database with sample data."""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create violations table
        cursor.execute(
            """
            CREATE TABLE violations (
                id INTEGER PRIMARY KEY,
                violation_type TEXT,
                description TEXT,
                vehicle_type TEXT,
                fine_amount TEXT,
                additional_penalty TEXT,
                legal_reference TEXT,
                keywords TEXT,
                document_id INTEGER
            )
        """
        )

        # Create legal_documents table
        cursor.execute(
            """
            CREATE TABLE legal_documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                code TEXT,
                content TEXT,
                effective_date TEXT,
                status TEXT
            )
        """
        )

        # Create health_advice table
        cursor.execute(
            """
            CREATE TABLE health_advice (
                id INTEGER PRIMARY KEY,
                category TEXT,
                condition_name TEXT,
                advice TEXT,
                keywords TEXT
            )
        """
        )

        # Create nutrition_data table
        cursor.execute(
            """
            CREATE TABLE nutrition_data (
                id INTEGER PRIMARY KEY,
                food_name TEXT,
                category TEXT,
                benefits TEXT,
                nutritional_value TEXT,
                recommended_for TEXT
            )
        """
        )

        # Create exercise_data table
        cursor.execute(
            """
            CREATE TABLE exercise_data (
                id INTEGER PRIMARY KEY,
                exercise_name TEXT,
                category TEXT,
                description TEXT,
                benefits TEXT,
                suitable_for TEXT
            )
        """
        )

        # Insert sample data
        sample_violations = [
            (
                1,
                "tốc độ",
                "Xe máy chạy quá tốc độ",
                "xe máy",
                "500.000 VND",
                "Tước bằng 1-3 tháng",
                "NĐ100",
                "tốc độ xe máy",
                1,
            ),
            (
                2,
                "nồng độ cồn",
                "Uống rượu lái xe",
                "xe máy",
                "6.000.000 VND",
                "Tước bằng 16-18 tháng",
                "NĐ100",
                "cồn rượu bia",
                1,
            ),
        ]

        cursor.executemany(
            "INSERT INTO violations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sample_violations,
        )

        sample_documents = [
            (
                1,
                "Nghị định 100/2019/NĐ-CP",
                "NĐ100",
                "Quy định xử phạt vi phạm hành chính về giao thông",
                "2020-01-01",
                "active",
            )
        ]

        cursor.executemany(
            "INSERT INTO legal_documents VALUES (?, ?, ?, ?, ?, ?)", sample_documents
        )

        # Insert sample health data
        sample_health_advice = [
            (
                1,
                "đau đầu",
                "Đau đầu thông thường",
                "Nghỉ ngơi, uống nhiều nước",
                "đau đầu",
            )
        ]

        cursor.executemany(
            "INSERT INTO health_advice VALUES (?, ?, ?, ?, ?)", sample_health_advice
        )

        sample_nutrition = [
            (
                1,
                "Chuối",
                "Trái cây",
                "Tốt cho tim mạch",
                "Kali, Vitamin B6",
                "Mọi lứa tuổi",
            )
        ]

        cursor.executemany(
            "INSERT INTO nutrition_data VALUES (?, ?, ?, ?, ?, ?)", sample_nutrition
        )

        sample_exercise = [
            (
                1,
                "Đi bộ",
                "Cardio",
                "Đi bộ 30 phút mỗi ngày",
                "Tốt cho tim mạch",
                "Mọi lứa tuổi",
            )
        ]

        cursor.executemany(
            "INSERT INTO exercise_data VALUES (?, ?, ?, ?, ?, ?)", sample_exercise
        )

        conn.commit()
        conn.close()

    def test_home_endpoint(self):
        """Test the home endpoint."""
        response = self.app.get("/")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("message", data)
        self.assertIn("endpoints", data)
        self.assertIn("traffic_law", data["endpoints"])
        self.assertIn("healthcare", data["endpoints"])

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            with patch("app.health_bot") as mock_health_bot:
                # Mock the violations and health data
                mock_traffic_bot.violations = [1, 2, 3]  # 3 violations
                mock_health_bot.health_advice = [1, 2]  # 2 health advice
                mock_health_bot.nutrition_data = [1]  # 1 nutrition data
                mock_health_bot.exercise_data = [1, 2, 3, 4]  # 4 exercise data
                mock_health_bot.emergency_conditions = [1]  # 1 emergency condition

                response = self.app.get("/health")

                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertEqual(data["status"], "healthy")
                self.assertEqual(data["traffic_law_violations_loaded"], 3)
                self.assertEqual(data["healthcare_data_loaded"]["health_advice"], 2)
                self.assertIn("timestamp", data)

    # Traffic Law Endpoints Tests
    def test_traffic_webhook_success(self):
        """Test traffic law webhook with valid request."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.generate_response.return_value = {
                "answer": "Test traffic response",
                "confidence": 0.8,
            }

            webhook_data = {"queryResult": {"queryText": "xe máy chạy quá tốc độ"}}

            response = self.app.post(
                "/traffic-law/webhook",
                data=json.dumps(webhook_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("fulfillmentText", data)
            self.assertEqual(data["fulfillmentText"], "Test traffic response")

    def test_traffic_webhook_empty_query(self):
        """Test traffic law webhook with empty query."""
        webhook_data = {"queryResult": {"queryText": ""}}

        response = self.app.post(
            "/traffic-law/webhook",
            data=json.dumps(webhook_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("Vui lòng nhập câu hỏi của bạn", data["fulfillmentText"])

    def test_traffic_chat_success(self):
        """Test traffic law chat endpoint."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.generate_response.return_value = {
                "answer": "Test traffic response",
                "confidence": 0.8,
                "violations_found": 1,
            }

            chat_data = {"message": "xe máy chạy quá tốc độ"}

            response = self.app.post(
                "/traffic-law/chat",
                data=json.dumps(chat_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("answer", data)
            self.assertIn("confidence", data)

    def test_traffic_chat_missing_message(self):
        """Test traffic law chat endpoint with missing message."""
        response = self.app.post(
            "/traffic-law/chat", data=json.dumps({}), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    # Healthcare Endpoints Tests
    def test_healthcare_webhook_success(self):
        """Test healthcare webhook with valid request."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.generate_health_response.return_value = {
                "answer": "Test health response",
                "confidence": 0.8,
            }

            webhook_data = {"queryResult": {"queryText": "đau đầu"}}

            response = self.app.post(
                "/healthcare/webhook",
                data=json.dumps(webhook_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("fulfillmentText", data)
            self.assertEqual(data["fulfillmentText"], "Test health response")

    def test_healthcare_webhook_empty_query(self):
        """Test healthcare webhook with empty query."""
        webhook_data = {"queryResult": {"queryText": ""}}

        response = self.app.post(
            "/healthcare/webhook",
            data=json.dumps(webhook_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("Vui lòng nhập câu hỏi sức khỏe của bạn", data["fulfillmentText"])

    def test_healthcare_chat_success(self):
        """Test healthcare chat endpoint."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.generate_health_response.return_value = {
                "answer": "Test health response",
                "confidence": 0.8,
                "results_found": 1,
            }

            chat_data = {"message": "đau đầu"}

            response = self.app.post(
                "/healthcare/chat",
                data=json.dumps(chat_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("answer", data)
            self.assertIn("confidence", data)

    def test_healthcare_chat_missing_message(self):
        """Test healthcare chat endpoint with missing message."""
        response = self.app.post(
            "/healthcare/chat", data=json.dumps({}), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    # Traffic Law Admin Endpoints Tests
    def test_add_traffic_violation_success(self):
        """Test adding traffic violation."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.add_violation.return_value = 123

            violation_data = {
                "violation_type": "test",
                "description": "Test violation",
                "vehicle_type": "xe máy",
                "fine_amount": "500.000 VND",
                "legal_reference": "Test law",
                "keywords": "test",
            }

            response = self.app.post(
                "/traffic-law/violations",
                data=json.dumps(violation_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["id"], 123)
            self.assertIn("message", data)

    def test_add_traffic_violation_no_data(self):
        """Test adding traffic violation without data."""
        response = self.app.post(
            "/traffic-law/violations",
            data=json.dumps({}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_update_traffic_violation_success(self):
        """Test updating traffic violation."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.update_violation.return_value = True

            update_data = {"description": "Updated description"}

            response = self.app.put(
                "/traffic-law/violations/1",
                data=json.dumps(update_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("message", data)

    def test_update_traffic_violation_not_found(self):
        """Test updating non-existent traffic violation."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.update_violation.return_value = False

            update_data = {"description": "Updated description"}

            response = self.app.put(
                "/traffic-law/violations/999",
                data=json.dumps(update_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 404)
            data = json.loads(response.data)
            self.assertIn("error", data)

    def test_delete_traffic_violation_success(self):
        """Test deleting traffic violation."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.delete_violation.return_value = True

            response = self.app.delete("/traffic-law/violations/1")

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("message", data)

    def test_delete_traffic_violation_not_found(self):
        """Test deleting non-existent traffic violation."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.delete_violation.return_value = False

            response = self.app.delete("/traffic-law/violations/999")

            self.assertEqual(response.status_code, 404)
            data = json.loads(response.data)
            self.assertIn("error", data)

    def test_add_traffic_violations_batch(self):
        """Test adding multiple traffic violations."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.add_violation.side_effect = [1, 2, 3]

            violations_data = [
                {"violation_type": "test1", "description": "Test 1"},
                {"violation_type": "test2", "description": "Test 2"},
                {"violation_type": "test3", "description": "Test 3"},
            ]

            response = self.app.post(
                "/traffic-law/violations/batch",
                data=json.dumps(violations_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["ids"], [1, 2, 3])
            self.assertIn("message", data)

    def test_add_traffic_violations_batch_invalid_data(self):
        """Test adding multiple traffic violations with invalid data."""
        response = self.app.post(
            "/traffic-law/violations/batch",
            data=json.dumps("not a list"),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    # Healthcare Admin Endpoints Tests
    def test_add_health_advice_success(self):
        """Test adding health advice."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.add_health_advice.return_value = 456

            advice_data = {
                "category": "test",
                "condition_name": "Test condition",
                "advice": "Test advice",
                "keywords": "test",
            }

            response = self.app.post(
                "/healthcare/health-advice",
                data=json.dumps(advice_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["id"], 456)
            self.assertIn("message", data)

    def test_add_nutrition_data_success(self):
        """Test adding nutrition data."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.add_nutrition_data.return_value = 789

            nutrition_data = {
                "food_name": "Test food",
                "category": "test",
                "benefits": "Test benefits",
                "nutritional_value": "Test value",
                "recommended_for": "Test group",
            }

            response = self.app.post(
                "/healthcare/nutrition-data",
                data=json.dumps(nutrition_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["id"], 789)
            self.assertIn("message", data)

    def test_add_exercise_data_success(self):
        """Test adding exercise data."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.add_exercise_data.return_value = 101

            exercise_data = {
                "exercise_name": "Test exercise",
                "category": "test",
                "description": "Test description",
                "benefits": "Test benefits",
                "suitable_for": "Test group",
            }

            response = self.app.post(
                "/healthcare/exercise-data",
                data=json.dumps(exercise_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["id"], 101)
            self.assertIn("message", data)

    # Error Handling Tests
    def test_traffic_webhook_exception(self):
        """Test traffic webhook with exception."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.generate_response.side_effect = Exception("Test error")

            webhook_data = {"queryResult": {"queryText": "test query"}}

            response = self.app.post(
                "/traffic-law/webhook",
                data=json.dumps(webhook_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn("fulfillmentText", data)
            self.assertIn("lỗi xảy ra", data["fulfillmentText"])

    def test_healthcare_chat_exception(self):
        """Test healthcare chat with exception."""
        with patch("app.health_bot") as mock_health_bot:
            mock_health_bot.generate_health_response.side_effect = Exception(
                "Test error"
            )

            chat_data = {"message": "test query"}

            response = self.app.post(
                "/healthcare/chat",
                data=json.dumps(chat_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn("error", data)

    def test_add_violation_exception(self):
        """Test adding violation with exception."""
        with patch("app.traffic_bot") as mock_traffic_bot:
            mock_traffic_bot.add_violation.side_effect = Exception("Database error")

            violation_data = {"violation_type": "test", "description": "Test violation"}

            response = self.app.post(
                "/traffic-law/violations",
                data=json.dumps(violation_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn("error", data)

    # Invalid JSON Tests
    def test_invalid_json_request(self):
        """Test request with invalid JSON."""
        response = self.app.post(
            "/traffic-law/chat",
            data="invalid json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 500)

    def test_missing_content_type(self):
        """Test request without proper content type."""
        response = self.app.post(
            "/traffic-law/chat", data=json.dumps({"message": "test"})
        )

        # Should still work but might not parse JSON correctly
        self.assertIn(response.status_code, [200, 400, 500])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
