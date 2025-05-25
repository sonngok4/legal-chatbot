import json
import sqlite3


def verify_migration():
    """Verify the migrated data"""
    # Connect to SQLite database
    conn = sqlite3.connect("legal_database.db")
    cursor = conn.cursor()

    # Get counts
    cursor.execute("SELECT COUNT(*) FROM legal_documents")
    doc_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM violations")
    violation_count = cursor.fetchone()[0]

    # Get sample data
    cursor.execute(
        """
        SELECT v.*, d.code as document_code
        FROM violations v
        JOIN legal_documents d ON v.document_id = d.id
        LIMIT 5
    """
    )
    sample_data = cursor.fetchall()

    # Print verification results
    print(f"Verification Results:")
    print(f"Total legal documents: {doc_count}")
    print(f"Total violations: {violation_count}")
    print("\nSample Data:")
    for row in sample_data:
        print(f"\nViolation ID: {row[0]}")
        print(f"Type: {row[1]}")
        print(f"Description: {row[2]}")
        print(f"Vehicle Type: {row[3]}")
        print(f"Fine Amount: {row[4]}")
        print(f"Legal Reference: {row[6]}")
        print(f"Document Code: {row[8]}")

    conn.close()


if __name__ == "__main__":
    verify_migration()
