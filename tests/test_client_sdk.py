from client.sdk import DocumentAnalystClient


def test_extract_answer_handles_list_payload() -> None:
    payload = [
        {
            "messages": [{"role": "ai", "content": "The net income was 1107 billion"}],
            "final_answer": "The net income was 1107 billion",
        }
    ]

    assert DocumentAnalystClient._extract_answer(payload) == "The net income was 1107 billion"
