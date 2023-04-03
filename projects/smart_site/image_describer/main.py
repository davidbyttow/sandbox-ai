import torch
from PIL import Image
from clip import clip


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
        similarity = (
            (100.0 * image_features.matmul(text_features.T))
            .softmax(dim=-1)
            .cpu()
            .numpy()
        )
    return similarity


def main():
    model, preprocess, device = load_clip_model()

    descriptions = [
        "minimalistic design",
        "dark theme",
        "bright and colorful theme",
        "intricate layout",
        "large, bold typography",
        "vibrant color scheme",
        "monochromatic design",
        "grid-based layout",
        "complex navigation",
        "simple navigation",
        # Add more descriptions as needed
    ]
    tokenized_descriptions = clip.tokenize(descriptions).to(device)

    while True:
        image_path = input(
            "Enter the path to the screenshot (or type 'exit' to quit): "
        )

        if image_path.lower() == "exit":
            break

        try:
            image_features = get_image_features(image_path, preprocess, device, model)
            similarity_scores = get_similarity_scores(
                image_features, tokenized_descriptions, device, model
            )
            top_description_index = similarity_scores.argmax()
            top_description = descriptions[top_description_index]
            print(
                f"The highest-scoring description for the screenshot is: '{top_description}'\n"
            )
        except Exception as e:
            print(f"Error processing the image: {e}\n")


if __name__ == "__main__":
    main()
