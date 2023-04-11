import json


def gen_anchor(name):
    return name.lower().replace(" ", "-")


def gen_md_link(name):
    return f"[{name}](#{gen_anchor(name)})"


def gen_href(name):
    return f'<a id="{gen_anchor(name)}"></a>'


content = "# Learning Plan\n\n"
with open("out.json", "r") as f:
    data = json.load(f)
    for subject in data:
        link = gen_md_link(subject["title"])
        content += f"* {link}\n"
        for topic in subject["topics"]:
            link = gen_md_link(topic["title"])
            content += f"  * {link}\n"
            for lesson in topic["lessons"]:
                link = gen_md_link(lesson["title"])
                content += f"    * {link}\n"
    content += "\n\n"
    for subject in data:
        aid = gen_href(subject["title"])
        content += f"## {aid}" + subject["title"] + "\n"
        content += subject["desc"] + "\n"
        content += "\n"
        for topic in subject["topics"]:
            aid = gen_href(topic["title"])
            content += f"### {aid}" + topic["title"] + "\n"
            content += topic["desc"] + "\n"
            content += "\n"
            for lesson in topic["lessons"]:
                aid = gen_href(subject["title"])
                content += f"#### {aid}" + lesson["title"] + "\n"
                if lesson["desc"]:
                    content += lesson["desc"] + "\n"
                content += "\n"
content += "\n"

print(content)
