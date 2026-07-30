from research_agent.tools.json_utils import extract_json_object


def test_extract_json_object_from_plain_json():
    assert extract_json_object('{"answer": 1}') == {"answer": 1}


def test_extract_json_object_from_json_code_fence():
    text = """```json
    {"answer": 1}
    ```"""

    assert extract_json_object(text) == {"answer": 1}


def test_extract_json_object_from_surrounding_text():
    text = 'Here is the result: {"answer": 1} Thanks.'

    assert extract_json_object(text) == {"answer": 1}
