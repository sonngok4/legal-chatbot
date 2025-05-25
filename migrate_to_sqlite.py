import json
import sqlite3
from datetime import datetime


def create_database():
    """Create SQLite database and tables"""
    conn = sqlite3.connect("legal_database.db")
    cursor = conn.cursor()

    # Create legal_documents table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS legal_documents (
        id INTEGER PRIMARY KEY,
        title TEXT,
        code TEXT,
        content TEXT,
        effective_date DATE,
        status TEXT DEFAULT 'active'
    )
    """
    )

    # Create violations table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY,
        violation_type TEXT,
        description TEXT,
        vehicle_type TEXT,
        fine_amount TEXT,
        additional_penalty TEXT,
        legal_reference TEXT,
        keywords TEXT,
        document_id INTEGER,
        FOREIGN KEY (document_id) REFERENCES legal_documents(id)
    )
    """
    )

    conn.commit()
    return conn


def migrate_data():
    """Migrate data from JSON to SQLite"""
    # Create database and tables
    conn = create_database()
    cursor = conn.cursor()

    # Read JSON data
    with open("violations.json", "r", encoding="utf-8") as f:
        violations_data = json.load(f)

    # Extract unique legal references
    legal_references = set()
    for violation in violations_data:
        if "legal_reference" in violation:
            legal_references.add(violation["legal_reference"])

    # Insert legal documents
    document_map = {}  # Map legal references to document IDs
    for ref in legal_references:
        # Extract code from reference (e.g., "Nghị định 100/2019")
        code = ref.split(",")[-1].strip()

        cursor.execute(
            """
        INSERT INTO legal_documents (title, code, content, effective_date, status)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                f"Văn bản {code}",
                code,
                f"Nội dung văn bản {code}",
                datetime.now().date(),  # You should update this with actual effective date
                "active",
            ),
        )
        document_map[ref] = cursor.lastrowid

    # Insert violations
    for violation in violations_data:
        document_id = document_map.get(violation.get("legal_reference", ""), None)

        cursor.execute(
            """
        INSERT INTO violations (
            violation_type, description, vehicle_type, fine_amount,
            additional_penalty, legal_reference, keywords, document_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                violation.get("violation_type", ""),
                violation.get("description", ""),
                violation.get("vehicle_type", ""),
                violation.get("fine_amount", ""),
                violation.get("additional_penalty", ""),
                violation.get("legal_reference", ""),
                violation.get("keywords", ""),
                document_id,
            ),
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    migrate_data()
    print("Migration completed successfully!")
