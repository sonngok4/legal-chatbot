import json
import os
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

from healthcare_chatbot import HealthCareChatbot
from logging_config import setup_logging
from traffic_law_chatbot import TrafficLawChatbot

app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logging()

# Initialize chatbots
logger.info("Initializing chatbots...")
traffic_bot = TrafficLawChatbot()
health_bot = HealthCareChatbot()
logger.info("Chatbots initialized successfully")


def create_dialogflow_response(fulfillment_text, session_id=None):
    """Create a standardized Dialogflow webhook response"""
    response = {
        "fulfillmentText": fulfillment_text,
        "fulfillmentMessages": [{"text": {"text": [fulfillment_text]}}],
        "source": "news-chatbot-api",
    }

    if session_id:
        response["session"] = session_id

    return response


def validate_dialogflow_request(req):
    """Validate Dialogflow webhook request"""
    if not req:
        return False, "Empty request body"

    if "queryResult" not in req:
        return False, "Missing queryResult in request"

    query_result = req.get("queryResult", {})
    if "queryText" not in query_result:
        return False, "Missing queryText in queryResult"

    return True, None


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "News Chatbot API is running!",
            "endpoints": {
                "traffic_law": {
                    "webhook": "/traffic-law/webhook (POST)",
                    "chat": "/traffic-law/chat (POST)",
                },
                "healthcare": {
                    "webhook": "/healthcare/webhook (POST)",
                    "chat": "/healthcare/chat (POST)",
                },
            },
        }
    )


# Traffic Law endpoints
@app.route("/traffic-law/webhook", methods=["POST"])
def traffic_webhook():
    """Webhook endpoint for Traffic Law Dialogflow agent"""
    try:
        req = request.get_json()
        is_valid, error_msg = validate_dialogflow_request(req)

        if not is_valid:
            logger.warning(f"Invalid Dialogflow request: {error_msg}")
            return jsonify(
                create_dialogflow_response(
                    "Tôi không hiểu câu hỏi của bạn. Vui lòng hỏi lại về luật giao thông đường bộ."
                )
            )

        query_result = req.get("queryResult", {})
        query_text = query_result.get("queryText", "")
        session_id = req.get("session", "")

        logger.info(
            f"Received traffic law webhook request - Session: {session_id}, Query: {query_text}"
        )

        result = traffic_bot.generate_response(query_text)

        # Create rich response with additional context if available
        response = create_dialogflow_response(result["answer"], session_id)

        # Add additional context if available
        if "context" in result:
            response["outputContexts"] = [
                {
                    "name": f"{session_id}/contexts/traffic-law-context",
                    "lifespanCount": 5,
                    "parameters": result["context"],
                }
            ]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in traffic webhook: {str(e)}")
        return jsonify(
            create_dialogflow_response(
                "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."
            )
        )


@app.route("/traffic-law/chat", methods=["POST"])
def traffic_chat():
    """Direct chat endpoint for Traffic Law"""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            logger.warning("Invalid request: missing message")
            return jsonify({"error": "Vui lòng gửi message trong request body"}), 400

        query = data.get("message", "").strip()
        logger.info(f"Received traffic law chat request: {query}")
        result = traffic_bot.generate_response(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in traffic chat: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Healthcare endpoints
@app.route("/healthcare/webhook", methods=["POST"])
def healthcare_webhook():
    """Webhook endpoint for Healthcare Dialogflow agent"""
    try:
        req = request.get_json()
        is_valid, error_msg = validate_dialogflow_request(req)

        if not is_valid:
            logger.warning(f"Invalid Dialogflow request: {error_msg}")
            return jsonify(
                create_dialogflow_response(
                    "Tôi không hiểu câu hỏi của bạn. Vui lòng hỏi lại về vấn đề sức khỏe."
                )
            )

        query_result = req.get("queryResult", {})
        query_text = query_result.get("queryText", "")
        session_id = req.get("session", "")

        logger.info(
            f"Received healthcare webhook request - Session: {session_id}, Query: {query_text}"
        )

        result = health_bot.generate_health_response(query_text)

        # Create rich response with additional context if available
        response = create_dialogflow_response(result["answer"], session_id)

        # Add additional context if available
        if "context" in result:
            response["outputContexts"] = [
                {
                    "name": f"{session_id}/contexts/healthcare-context",
                    "lifespanCount": 5,
                    "parameters": result["context"],
                }
            ]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in healthcare webhook: {str(e)}")
        return jsonify(
            create_dialogflow_response(
                "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."
            )
        )


@app.route("/healthcare/chat", methods=["POST"])
def healthcare_chat():
    """Direct chat endpoint for Healthcare"""
    try:
        data = request.get_json()
        print(data)
        if not data or "message" not in data:
            logger.warning("Invalid request: missing message")
            return jsonify({"error": "Vui lòng gửi message trong request body"}), 400

        query = data.get("message", "").strip()
        logger.info(f"Received healthcare chat request: {query}")
        result = health_bot.generate_health_response(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in healthcare chat: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "traffic_law_violations_loaded": len(traffic_bot.violations),
            "healthcare_data_loaded": {
                "health_advice": len(health_bot.health_advice),
                "nutrition_data": len(health_bot.nutrition_data),
                "exercise_data": len(health_bot.exercise_data),
                "emergency_conditions": len(health_bot.emergency_conditions),
            },
            "timestamp": str(datetime.now()),
        }
    )


# Admin endpoints for Traffic Law data management
@app.route("/traffic-law/violations", methods=["POST"])
def add_traffic_violation():
    """Add new traffic violation"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Invalid request: no data provided")
            return jsonify({"error": "No data provided"}), 400

        logger.info("Adding new traffic violation")
        violation_id = traffic_bot.add_violation(data)
        return jsonify({"id": violation_id, "message": "Violation added successfully"})
    except Exception as e:
        logger.error(f"Error adding violation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/violations/<int:violation_id>", methods=["PUT"])
def update_traffic_violation(violation_id):
    """Update traffic violation"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        success = traffic_bot.update_violation(violation_id, data)
        if success:
            return jsonify({"message": "Violation updated successfully"})
        return jsonify({"error": "Violation not found"}), 404
    except Exception as e:
        logger.error(f"Error updating violation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/violations/<int:violation_id>", methods=["DELETE"])
def delete_traffic_violation(violation_id):
    """Delete traffic violation"""
    try:
        success = traffic_bot.delete_violation(violation_id)
        if success:
            return jsonify({"message": "Violation deleted successfully"})
        return jsonify({"error": "Violation not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting violation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/violations/batch", methods=["POST"])
def add_traffic_violations_batch():
    """Add multiple traffic violations at once"""
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            logger.warning("Invalid request: data must be a list of violations")
            return jsonify({"error": "Data must be a list of violations"}), 400

        logger.info(f"Adding {len(data)} new traffic violations")
        violation_ids = []

        for violation_data in data:
            violation_id = traffic_bot.add_violation(violation_data)
            violation_ids.append(violation_id)

        return jsonify(
            {
                "ids": violation_ids,
                "message": f"Successfully added {len(violation_ids)} violations",
            }
        )
    except Exception as e:
        logger.error(f"Error adding violations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/legal-documents", methods=["POST"])
def add_traffic_legal_document():
    """Add new legal document for traffic law"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        doc_id = traffic_bot.add_legal_document(data)
        return jsonify({"id": doc_id, "message": "Legal document added successfully"})
    except Exception as e:
        logger.error(f"Error adding legal document: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/legal-documents/<int:doc_id>", methods=["PUT"])
def update_traffic_legal_document(doc_id):
    """Update legal document for traffic law"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        success = traffic_bot.update_legal_document(doc_id, data)
        if success:
            return jsonify({"message": "Legal document updated successfully"})
        return jsonify({"error": "Legal document not found"}), 404
    except Exception as e:
        logger.error(f"Error updating legal document: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/traffic-law/legal-documents/<int:doc_id>", methods=["DELETE"])
def delete_traffic_legal_document(doc_id):
    """Delete legal document for traffic law"""
    try:
        success = traffic_bot.delete_legal_document(doc_id)
        if success:
            return jsonify({"message": "Legal document deleted successfully"})
        return jsonify({"error": "Legal document not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting legal document: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Admin endpoints for Healthcare data management
@app.route("/healthcare/health-advice", methods=["POST"])
def add_health_advice():
    """Add new health advice"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        advice_id = health_bot.add_health_advice(data)
        return jsonify({"id": advice_id, "message": "Health advice added successfully"})
    except Exception as e:
        logger.error(f"Error adding health advice: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/healthcare/nutrition-data", methods=["POST"])
def add_nutrition_data():
    """Add new nutrition data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        nutrition_id = health_bot.add_nutrition_data(data)
        return jsonify(
            {"id": nutrition_id, "message": "Nutrition data added successfully"}
        )
    except Exception as e:
        logger.error(f"Error adding nutrition data: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/healthcare/exercise-data", methods=["POST"])
def add_exercise_data():
    """Add new exercise data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        exercise_id = health_bot.add_exercise_data(data)
        return jsonify(
            {"id": exercise_id, "message": "Exercise data added successfully"}
        )
    except Exception as e:
        logger.error(f"Error adding exercise data: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
