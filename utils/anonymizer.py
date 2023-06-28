import pdfplumber
import re
import json
import typing
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def extract_text(filepath: str) -> str:
    # Open the file and create a pdf object.
    pdf = pdfplumber.open(filepath)
    all_text = ""
    # Iterate over each page and extract the text of each page.
    for number, pageText in enumerate(pdf.pages):
        all_text += "\n" + pageText.extract_text()
    return all_text


def first_filter(input_string):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(input_string)
    for result in ner_results:
        start_index, end_index = result['start'], result['end']
        mask = '*'
        word_length = end_index - start_index
        full_mask = mask * word_length
        input_string = full_mask.join([input_string[:start_index], input_string[end_index:]])
    return input_string


def _get_personal_data(candidate_id: int) -> dict:
    # Mocking the response from https://subdomain.sage.hr/api/recruitment/applicants/{candidate_id}
    mock_response_filename = f"resources/example_data{candidate_id}.json"
    with open(mock_response_filename, "r") as read_file:
        personal_data = read_file.read()
    return json.loads(personal_data)


def _get_keywords(personal_data) -> typing.List[str]:
    keyword_list = []
    for field, field_value in personal_data["data"].items():
        if field in ["full_name", "first_name", "last_name", "email", "address", "phone_number"]:
            if " " in str(field_value):
                keyword_list += field_value.split()
            elif field_value is not None and field_value != {}:
                keyword_list.append(field_value)
    return keyword_list


def final_filter(candidate_id: int, text: str) -> str:
    personal_data = _get_personal_data(candidate_id)
    keyword_list = _get_keywords(personal_data)
    for keyword in keyword_list:
        escaped_keyword = re.escape(keyword)
        text = re.sub(escaped_keyword, "*****", text, flags=re.IGNORECASE)
    return text


def anonymize(filename):
    text = extract_text(filename)
    mock_candidate_id = filename.split('.pdf')[-2][-1]  # Get mock candidate id from filename
    filtered_text = first_filter(text)
    output = final_filter(candidate_id=mock_candidate_id, text=filtered_text)
    return output