"""Integration test for the agent service using the configured Kilo model."""
import os
import sys

# Load .env before any app imports
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.service import create_agent_service
from app.schemas.agent import AgentRunRequest


def test_simple_question_no_tools():
    """Test that the agent can answer a simple question without tools."""
    service = create_agent_service()
    request = AgentRunRequest(message="Say hello in one short sentence.")
    response = service.run(request)
    print(f"Final answer: {response.final_answer}")
    print(f"Tool calls: {len(response.tool_calls)}")
    print(f"Rounds used: {response.rounds_used}")
    assert len(response.final_answer) > 0
    print("PASS: Simple question answered")


def test_count_requirements_integration():
    """Test that the agent calls count_requirements tool."""
    service = create_agent_service()
    request = AgentRunRequest(
        message=(
            "I have the following requirements: "
            "1. Real-time data ingestion, "
            "2. Identity resolution, "
            "3. GDPR compliance, "
            "4. API access, "
            "5. Support for 10 million records. "
            "How many requirements are there?"
        )
    )
    response = service.run(request)
    print(f"Final answer: {response.final_answer}")
    print(f"Tool calls: {len(response.tool_calls)}")
    print(f"Rounds used: {response.rounds_used}")
    for tc in response.tool_calls:
        print(f"  Tool: {tc.tool_name}, args: {tc.arguments}, result: {tc.result}")
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0].tool_name == "count_requirements"
    assert response.tool_calls[0].result == 5
    print("PASS: count_requirements tool called")


def test_calculate_total_integration():
    """Test that the agent calls calculate_total tool."""
    service = create_agent_service()
    request = AgentRunRequest(
        message=(
            "Calculate the total cost of: "
            "Widget A (quantity 3, unit price $10.00), "
            "Widget B (quantity 2, unit price $25.00)."
        )
    )
    response = service.run(request)
    print(f"Final answer: {response.final_answer}")
    print(f"Tool calls: {len(response.tool_calls)}")
    print(f"Rounds used: {response.rounds_used}")
    for tc in response.tool_calls:
        print(f"  Tool: {tc.tool_name}, args: {tc.arguments}, result: {tc.result}")
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0].tool_name == "calculate_total"
    assert response.tool_calls[0].result == 80.0
    print("PASS: calculate_total tool called")


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST: Kilo Tool Calling")
    print("=" * 60)

    try:
        test_simple_question_no_tools()
        print()
        test_count_requirements_integration()
        print()
        test_calculate_total_integration()
        print()
        print("=" * 60)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
    except Exception as exc:
        print(f"\nINTEGRATION TEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)