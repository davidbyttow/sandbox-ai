import os
import openai
import torch
from PIL import Image
from clip import clip

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def get_image_features(image_path, preprocess, device, model):
    image = Image.open(image_path).convert("RGB")
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
    return image_features


def get_similarity_scores(image_features, tokenized_descriptions, device, model):
    with torch.no_grad():
        text_features = model.encode_text(tokenized_descriptions)
        similarity = (100.0 * image_features.matmul(text_features.T)
                      ).softmax(dim=-1).cpu().numpy()
    return similarity


def generate_html(prompt, debug):
    with open('start.html', 'r') as f:
        start_html = f.read()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Given this initial HTML structure:\n\n{start_html}\n\nPlease modify the HTML and CSS styles to reflect a `{prompt}` aesthetic. Keep the basic structure and content, but update the colors, typography, and other design elements to match the desired style. Start your response with 'Modified HTML:' followed by the modified HTML and CSS, and then add 'Changes made:' followed by a brief description of the changes made:\n",
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    html_and_description = response.choices[0].text.strip()

    if debug:
        print("GPT Output ==========\n")
        print(f"\n{html_and_description}\n")
        print("End GPT Output ==========\n")

    # Find the positions of 'Modified HTML:' and 'Changes made:' while ignoring capitalization
    modified_html_pos = html_and_description.lower().find('modified html:')
    changes_made_pos = html_and_description.lower().find('changes made:')

    if modified_html_pos != -1 and changes_made_pos != -1:
        # Split the html_and_description into HTML code and description
        html_code = html_and_description[modified_html_pos +
                                         len('Modified HTML:'):changes_made_pos].strip()
        description = html_and_description[changes_made_pos +
                                           len('Changes made:'):].strip()

        # Write the modified HTML to the output file
        with open('output.html', 'w') as f:
            f.write(html_code.strip())

        # Print the description for the user
        print("\nDescription of changes:")
        print(description.strip())
    else:
        print("GPT did not generate the expected output. Please try again.")


def main():
    debug = False
    model, preprocess, device = load_clip_model()

    descriptions = [
        "minimalistic design with clean lines and plenty of whitespace",
        "dark theme with black or dark gray backgrounds and contrasting light text",
        "bright and colorful theme with vibrant background colors and bold, playful typography",
        "intricate layout with detailed design elements and complex visuals",
        "large, bold typography with strong headings and clear legibility",
        "vibrant color scheme with high contrast colors and a lively appearance",
        "monochromatic design with a single color palette and subtle variations in shade",
        "grid-based layout with content organized into a structured grid format",
        "complex navigation with multiple menus, dropdowns, or nested links",
        "simple navigation with a clean and intuitive menu structure"
    ]
    tokenized_descriptions = clip.tokenize(descriptions).to(device)

    print("This program uses GPT to generate the HTML for a basic blog page, relying on a provided screenshot of a website for aesthetic inspiration.\n\n")

    while True:
        image_path = input(
            "Enter the path to the screenshot of the website (or type 'exit' to quit): ")

        if image_path.lower() == 'exit':
            break

        try:
            image_features = get_image_features(
                image_path, preprocess, device, model)
            similarity_scores = get_similarity_scores(
                image_features, tokenized_descriptions, device, model)
            top_description_index = similarity_scores.argmax()
            top_description = descriptions[top_description_index]
            print(f"Chosen description: {top_description}\n")

            # Just call the function without assigning the result to a variable
            generate_html(top_description, debug)
            print("HTML code written to output.html.")

        except Exception as e:
            print(f"Error processing the image: {e}")


if __name__ == "__main__":
    main()
