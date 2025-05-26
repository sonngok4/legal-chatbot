import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from traffic_law_chatbot import TrafficLawChatbot


class TestTrafficLawChatbot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary database for testing
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db_path = self.test_db.name
        self.test_db.close()

        # Create test database schema and data
        self.setup_test_database()

        # Mock the database path and initialization
        with patch.object(TrafficLawChatbot, "__init__", return_value=None):
            self.chatbot = TrafficLawChatbot()
            self.chatbot.db_path = self.test_db_path
            self.chatbot.vectorizer = MagicMock()
            self.chatbot.violations = pd.DataFrame()
            self.chatbot.legal_documents = pd.DataFrame()
            self.chatbot.corpus = []
            self.chatbot.tfidf_matrix = None

    def tearDown(self):
        """Clean up after each test method."""
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

        # Insert sample data
        sample_violations = [
            (
                1,
                "tốc độ",
                "Xe máy chạy quá tốc độ 60-80km/h",
                "xe máy",
                "400.000 - 600.000 VND",
                "Tước bằng lái 1-3 tháng",
                "Nghị định 100/2019/NĐ-CP",
                "tốc độ xe máy",
                1,
            ),
            (
                2,
                "nồng độ cồn",
                "Điều khiển xe máy có nồng độ cồn",
                "xe máy",
                "6.000.000 - 8.000.000 VND",
                "Tước bằng lái 16-18 tháng",
                "Nghị định 100/2019/NĐ-CP",
                "cồn rượu bia xe máy",
                1,
            ),
            (
                3,
                "đèn đỏ",
                "Vượt đèn đỏ",
                "tất cả",
                "4.000.000 - 6.000.000 VND",
                "Tước bằng lái 1-3 tháng",
                "Nghị định 100/2019/NĐ-CP",
                "đèn đỏ tín hiệu",
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

        conn.commit()
        conn.close()

    def test_get_db_connection(self):
        """Test database connection."""
        conn = self.chatbot.get_db_connection()
        self.assertIsNotNone(conn)
        conn.close()

    def test_load_data(self):
        """Test data loading from database."""
        self.chatbot.load_data()

        self.assertIsInstance(self.chatbot.violations, pd.DataFrame)
        self.assertIsInstance(self.chatbot.legal_documents, pd.DataFrame)
        self.assertEqual(len(self.chatbot.violations), 3)
        self.assertEqual(len(self.chatbot.legal_documents), 1)
        self.assertEqual(len(self.chatbot.corpus), 3)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        test_cases = [
            ("Xe gắn máy chạy nhanh", "xe máy tốc độ"),
            ("Ô tô con vượt đèn đỏ", "ô tô đèn đỏ"),
            ("Uống rượu lái xe", "nồng độ cồn lái xe"),
            ("Không có bằng lái", "không có giấy phép lái xe"),
        ]

        for input_text, expected in test_cases:
            result = self.chatbot.preprocess_text(input_text)
            self.assertIn(expected.replace(" ", ""), result.replace(" ", ""))

    def test_extract_entities(self):
        """Test entity extraction from queries."""
        test_cases = [
            {
                "query": "xe máy chạy 80km/h",
                "expected": {
                    "vehicle_type": "xe máy",
                    "violation_types": ["tốc độ"],
                    "speed": 80,
                },
            },
            {
                "query": "ô tô uống rượu",
                "expected": {
                    "vehicle_type": "ô tô",
                    "violation_types": ["nồng độ cồn"],
                },
            },
            {"query": "vượt đèn đỏ", "expected": {"violation_types": ["đèn đỏ"]}},
        ]

        for case in test_cases:
            entities = self.chatbot.extract_entities(case["query"])
            for key, expected_value in case["expected"].items():
                if key in entities:
                    if isinstance(expected_value, list):
                        self.assertEqual(entities[key], expected_value)
                    else:
                        self.assertEqual(entities[key], expected_value)

    @patch("traffic_law_chatbot.TfidfVectorizer")
    @patch("traffic_law_chatbot.cosine_similarity")
    def test_search_violations(self, mock_cosine, mock_vectorizer):
        """Test violation search functionality."""
        # Setup mocks
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vectorizer_instance.transform.return_value = MagicMock()

        # Mock cosine similarity to return a numpy array
        mock_cosine.return_value = np.array([[0.8, 0.6, 0.3]])

        # Load test data
        self.chatbot.load_data()
        self.chatbot.vectorizer = mock_vectorizer_instance
        self.chatbot.tfidf_matrix = MagicMock()

        # Test search
        results = self.chatbot.search_violations("xe máy chạy nhanh")

        self.assertIsInstance(results, list)
        mock_cosine.assert_called_once()

    def test_generate_response_empty_query(self):
        """Test response generation with empty query."""
        response = self.chatbot.generate_response("")

        self.assertIn("answer", response)
        self.assertEqual(response["confidence"], 0.0)
        self.assertIn("Vui lòng nhập câu hỏi", response["answer"])

    def test_generate_response_no_results(self):
        """Test response generation when no violations found."""
        # Mock search to return empty results
        with patch.object(self.chatbot, "search_violations", return_value=[]):
            response = self.chatbot.generate_response("query with no results")

            self.assertIn("answer", response)
            self.assertEqual(response["confidence"], 0.0)
            self.assertIn("không tìm thấy thông tin", response["answer"])

    def test_generate_response_with_results(self):
        """Test response generation with search results."""
        # Mock search results
        mock_violations = [
            {
                "description": "Test violation",
                "vehicle_type": "xe máy",
                "fine_amount": "500.000 VND",
                "additional_penalty": "Test penalty",
                "legal_reference": "Test law",
                "confidence": 0.9,
            }
        ]

        with patch.object(
            self.chatbot, "search_violations", return_value=mock_violations
        ):
            with patch.object(self.chatbot, "extract_entities", return_value={}):
                response = self.chatbot.generate_response("test query")

                self.assertIn("answer", response)
                self.assertGreater(response["confidence"], 0)
                self.assertIn("Test violation", response["answer"])

    def test_add_violation(self):
        """Test adding new violation."""
        violation_data = {
            "violation_type": "test",
            "description": "Test violation",
            "vehicle_type": "test vehicle",
            "fine_amount": "1.000.000 VND",
            "legal_reference": "Test law",
            "keywords": "test keywords",
        }

        with patch.object(self.chatbot, "load_data"):
            violation_id = self.chatbot.add_violation(violation_data)
            self.assertIsInstance(violation_id, int)
            self.assertGreater(violation_id, 0)

    def test_update_violation(self):
        """Test updating existing violation."""
        update_data = {
            "description": "Updated description",
            "fine_amount": "2.000.000 VND",
        }

        with patch.object(self.chatbot, "load_data"):
            success = self.chatbot.update_violation(1, update_data)
            self.assertTrue(success)

    def test_delete_violation(self):
        """Test deleting violation."""
        with patch.object(self.chatbot, "load_data"):
            success = self.chatbot.delete_violation(1)
            self.assertTrue(success)

    def test_add_legal_document(self):
        """Test adding new legal document."""
        document_data = {
            "title": "Test Document",
            "code": "TEST001",
            "content": "Test content",
            "effective_date": "2024-01-01",
        }

        with patch.object(self.chatbot, "load_data"):
            doc_id = self.chatbot.add_legal_document(document_data)
            self.assertIsInstance(doc_id, int)
            self.assertGreater(doc_id, 0)

    def test_update_legal_document(self):
        """Test updating legal document."""
        update_data = {"title": "Updated Title", "content": "Updated content"}

        with patch.object(self.chatbot, "load_data"):
            success = self.chatbot.update_legal_document(1, update_data)
            self.assertTrue(success)

    def test_delete_legal_document(self):
        """Test deleting legal document."""
        with patch.object(self.chatbot, "load_data"):
            success = self.chatbot.delete_legal_document(1)
            self.assertTrue(success)

    def test_cached_search(self):
        """Test cached search functionality."""
        with patch.object(
            self.chatbot, "search_violations", return_value=[]
        ) as mock_search:
            # First call
            result1 = self.chatbot.cached_search("test query")
            # Second call should use cache
            result2 = self.chatbot.cached_search("test query")

            # Should only call search_violations once due to caching
            mock_search.assert_called_once()
            self.assertEqual(result1, result2)

    def test_learn_from_feedback(self):
        """Test feedback learning functionality."""
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.chatbot.learn_from_feedback("test query", True)
            mock_file.assert_called_once_with("feedback.log", "a", encoding="utf-8")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
