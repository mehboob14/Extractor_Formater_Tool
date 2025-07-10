from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
import shutil
from scraper import run_scrape_and_save, DATA_DIR, DOWNLOADED_IMAGES_DIR
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = Flask(__name__)

STATIC_IMAGE_DIR = os.path.join("static", "images")
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

OPENAI_API_KEY =os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4.1", temperature=0)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if os.path.exists(STATIC_IMAGE_DIR):
            shutil.rmtree(STATIC_IMAGE_DIR)

        os.makedirs(DOWNLOADED_IMAGES_DIR, exist_ok=True)
        os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

        dress_url = request.form.get("dress_url")
        if dress_url:
            run_scrape_and_save(dress_url)
            return redirect(url_for("output"))
    return render_template("index.html")


@app.route("/output")
def output():
    def read_file(path):
        try:
            with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    editors_notes = read_file("editors_notes.txt")
    size_fit = read_file("size_fit.txt")
    model_measurements = read_file("model_measurements.txt")
    details_care = read_file("details_care.txt")

    try:
        with open(os.path.join(DATA_DIR, "Size_guide.json"), "r", encoding="utf-8") as f:
            size_guide = json.load(f)
            size_headers = list(next(iter(size_guide.values())).keys())
    except:
        size_guide = {}
        size_headers = []

    image_files = []
    if os.path.exists(DOWNLOADED_IMAGES_DIR):
        for old in os.listdir(STATIC_IMAGE_DIR):
            os.remove(os.path.join(STATIC_IMAGE_DIR, old))

        image_files = [f for f in os.listdir(DOWNLOADED_IMAGES_DIR) if f.endswith(".jpeg")]
        for file in image_files:
            src = os.path.join(DOWNLOADED_IMAGES_DIR, file)
            dst = os.path.join(STATIC_IMAGE_DIR, file)
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())

    return render_template(
        "output.html",
        editors_notes=editors_notes,
        size_fit=size_fit,
        model_measurements=model_measurements,
        details_care=details_care,
        size_guide=size_guide,
        size_headers=size_headers,
        images=image_files
    )


@app.route("/format_prompts", methods=["POST"])
def format_prompts():
    try:
        responses = []
        index = 0

        def load_text(filename):
            path = os.path.join(DATA_DIR, filename)
            return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

        field_map = {
            "includes_images": lambda: "",
            "editors_notes": lambda: load_text("editors_notes.txt"),
            "details_care": lambda: load_text("details_care.txt"),
            "size_fit": lambda: load_text("size_fit.txt"),
            "model_measurements": lambda: load_text("model_measurements.txt"),
            "sizing_guide": lambda: json.dumps(
                json.load(open(os.path.join(DATA_DIR, "Size_guide.json"), "r", encoding="utf-8")),
                indent=2
            ) if os.path.exists(os.path.join(DATA_DIR, "Size_guide.json")) else ""
        }

        while True:
            prompt_key = f"prompt_{index}"
            fields_key = f"fields_{index}"

            if prompt_key not in request.form or fields_key not in request.form:
                break

            prompt_text = request.form.get(prompt_key, "").strip()
            selected_fields = json.loads(request.form.get(fields_key, "[]"))
            field_content = "\n\n".join([
                f"{field.replace('_', ' ').title()}:\n{field_map[field]()}"
                for field in selected_fields if field in field_map
            ])

            image_mapping_str = ""
            image_map = {}

            if "includes_images" in selected_fields:
                user_labels = []
                label_index = 0
                print(f"Image label keys for prompt {index}: {[k for k in request.form.keys() if k.startswith(f'image_label_{index}_')]}")
                while True:
                    key = f"image_label_{index}_{label_index}"
                    val = request.form.get(key)
                    if val is None:
                        break
                    if val.strip():
                        user_labels.append(val.strip())
                    label_index += 1

                print(user_labels)
                image_dir = os.path.join("static", "images")
                image_files = sorted([
                    f for f in os.listdir(image_dir)
                    if os.path.isfile(os.path.join(image_dir, f))
                ])

                for i, label in enumerate(user_labels):
                    if i < len(image_files):
                        path = os.path.join(image_dir, image_files[i]).replace("\\", "/")
                        image_map[label] = path

                if image_map:
                    image_mapping_str = "Image Mapping:\n" + "\n".join(
                        [f"{label} => {path}" for label, path in image_map.items()]
                    )

            full_prompt = f"{field_content}\n\nUser Prompt:\n{prompt_text}"
            if image_mapping_str:
                full_prompt += f"\n\n{image_mapping_str}\n\nUse the mapping above to associate the correct image with each label."

            messages = [
                SystemMessage(content=""),
                HumanMessage(content=full_prompt)
            ]

            llm_response = llm.invoke(messages)

            # Append the response inside the loop
            responses.append({
                "input_prompt": prompt_text,
                "fields": selected_fields,
                "response": llm_response.content,
                "images": image_map if image_map else {},
                "full_prompt": full_prompt
            })

            index += 1

        return jsonify({"results": responses})

    except Exception as e:
        print(f"LLM error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
