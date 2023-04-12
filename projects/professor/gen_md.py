import json
import sys


def gen_anchor(name):
    return name.lower().replace(" ", "-")


def gen_md_link(name):
    return f"[{name}](#{gen_anchor(name)})"


def gen_href(name):
    return f"<a id='{gen_anchor(name)}'></a>"


def gen_markdown(data, goal: str, reason: str, knowledge: str):
    content = "# Learning Hub\n\n"

    content += "Context:\n"
    content += f"* Stated goal: {goal}\n"
    content += f"* Reason: {reason}\n"
    content += f"* Existing knowledge: {knowledge}\n\n"

    content += "***\n"
    for subject in data:
        link = gen_md_link(subject["title"])
        content += f"* {link}\n"
        for topic in subject["topics"]:
            link = gen_md_link(topic["title"])
            content += f"  * {link}\n"
            for subtopic in topic["subtopics"]:
                if "tasks" in subtopic:
                    link = gen_md_link(subtopic["title"])
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
            for subtopic in topic["subtopics"]:
                if "tasks" in subtopic:
                    aid = gen_href(subtopic["title"])
                    content += f"#### {aid}" + subtopic["title"] + "\n"
                    content += subtopic["desc"] + "\n"
                    for task in subtopic["tasks"]:
                        content += f"- [ ] " + task["title"] + "\n"
                        if "desk" in task:
                            content += task["desc"] + "\n"
                        content += "\n"
                else:
                    content += f"* " + subtopic["title"] + "\n"
                content += "\n"
    return content.strip() + "\n"


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
        print(gen_markdown(data, "", "", ""))
