import openai
import os
import re
from flask import Flask, render_template, request, jsonify

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

debug = True

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_feedback', methods=['POST'])
def get_feedback():
    data = request.get_json()
    feedback = data.get('feedback')
    doc = data.get('doc')
    # html = data.get('html')
    # css = data.get('css')
    # js = data.get('js')

    # updated_page = process_feedback(feedback, html, css, js)
    updated_page = process_feedback_doc(feedback, doc)
    return jsonify(updated_page)

def process_feedback_doc(feedback, doc):
    # Define a prompt for GPT-3 based on the feedback, and the entire doc
    prompt = f"""
User Feedback: {feedback}
Current DOC: {doc}

Read the `User Feedback` and change the `Current DOC` based on the feedback.
Note this is the entire page, and includes embedded CSS and JS. Update it with the feedback, ensuring it still is valid.
Return the result in the following format (replace the <INSERT HERE> placeholder):

Updated DOC:<INSERT UPDATED DOC HERE>
"""

    # Call the GPT-3 API to get the response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    if debug:
        print(f'Prompt: {prompt}')
        print(f'Response: {response}')

    content = response.choices[0].text.strip().split("\n")

    try:
        updated_doc = content[content.index("Updated DOC:") + 1]
    except ValueError:
        updated_doc = doc # return old one

    return {
        'doc': updated_doc
    }

# def process_feedback(feedback, html, css, js):
#     # Define a prompt for GPT-3 based on the feedback, current HTML, CSS, and JavaScript
#     prompt = f"""
# User Feedback: {feedback}
# Current HTML: {html}
# Current CSS: {css}
# Current JS: {js}

# Read the `User Feedback` and change the `Current HTML`, `Current CSS`, and/or `Current JS` based on the feedback.
# Take these values and return the result in the following format (replace the <INSERT HERE> placeholders):

# Updated HTML:
# <INSERT UPDATED HTML HERE>

# Updated CSS:
# <INSERT UPDATED CSS HERE>

# Updated JS:
# <INSERT UPDATED JS HERE>
# """

#     # Call the GPT-3 API to get the response
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=2000,
#         n=1,
#         stop=None,
#         temperature=0.7,
#     )

#     if debug:
#         print(f'Prompt: {prompt}')
#         print(f'Response: {response}')

#     content = response.choices[0].text.strip().split("\n")

#     try:
#         updated_html = content[content.index("Updated HTML:") + 1]
#     except ValueError:
#         updated_html = html

#     try:
#         updated_css = content[content.index("Updated CSS:") + 1]
#     except ValueError:
#         updated_css = css

#     try:
#         updated_js = content[content.index("Updated JS:") + 1]
#     except ValueError:
#         updated_js = js

#     return {
#         'html': updated_html,
#         'css': updated_css,
#         'js': updated_js
#     }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
