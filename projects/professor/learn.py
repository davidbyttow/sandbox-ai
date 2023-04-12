import asyncio
from typing import Dict, List, Any
from gen_md import gen_markdown

from pydantic import BaseModel, Field
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
import langchain

print(langchain.__file__, langchain.__version__)

# TODO(d): refactor this to recursively build topics/subtopics and simplify the code

TEMPERATURE = 0
EMBEDDING_SIZE = 1536
VERBOSE = True
# MODEL = "gpt-4"
MODEL = "gpt-3.5-turbo"

TOPIC = "welding"
GOAL = "I want to learn how to mig weld within a few days"
REASON = "I want to be able to weld metal together for my professional projects"
KNOWLEDGE = "I am a professional cabinet maker and I am familiar with tig welding. I need to learn how to mig weld for my projects."


# TOPIC = "neural_networks"
# GOAL = ("I want to learn how to code neural networks.",)
# REASON = (
#     "I want to be able to code FCN, FNN, CNNs and GANs by hand using pytorch, but also understand the math.",
# )
# KNOWLEDGE = (
#     "I am an expert in python already and have a rough knowledge of linear algebra and statistics.",
# )

# TOPIC = "golf"
# GOAL = "I want to learn how to golf"
# REASON = "I want to be able to play golf with my friends on the weekends"
# KNOWLEDGE = "I know very little about golf. I've only caddyed once."


# TOPIC = "chess"
# GOAL = "I want to become a 2000 ELO rated chess player"
# REASON = "I want to be able to beat all of my friends and compete in tournaments"
# KNOWLEDGE = "I am an intermediate chess player, rated at 1200 so I already know the basics and tactics."


embeddings_model = OpenAIEmbeddings()


def prelude_context(goal: str, reason: str, knowledge: str):
    return (
        "You are a professor and tutor. You are generating a learning plan for me as your private student. I want to learn something efficiently and effectively, but have a fixed amount of time they'd like to do so.\n"
        f"Here is my goal (GOAL): {goal}\n"
        f"Here is why I want to do this (REASON): {reason}\n"
        f"Here is what I already know related to the topic (KNOWLEDGE): {knowledge}\n\n"
    )


def enumerate_items(items: List[Dict]) -> str:
    return "\n".join([f"{s['id']}. {s['title']}" for s in items]) + "\n"


def subjects_context(subjects: List[Dict]) -> str:
    return (
        "Here are the high-level subjects you previously told me to focus on:\n"
        + enumerate_items(subjects)
        + "\n\n"
    )


def topics_context(subject_id: int, subject_title: str, topics: List[Dict]) -> str:
    return (
        f'And for the subject #{subject_id} "{subject_title}", here are the topics you came up with for me to learn:\n'
        + enumerate_items(topics)
        + "\n\n"
    )


def subtopics_context(topic_id: int, topic_title: str, subtopics: List[Dict]) -> str:
    return (
        f'And for the topic #{topic_id} "{topic_title}", here are the subtopics you came up with for me to learn:\n'
        + enumerate_items(subtopics)
        + "\n\n"
    )


def split_at(text: str, sep: str) -> tuple:
    index = text.find(sep)
    return (text[:index], text[index + 1 :]) if index != -1 else (text, "")


def parse_items(response: str, split_desc=True) -> List[Dict]:
    items = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        num, content = split_at(line, ".")
        if not content:
            print(f"skipped response line: {line}")
            continue
        try:
            id = int(num)
        except ValueError:
            print(f"skipped response line: {line}")
            continue
        id = int(num)
        if split_desc:
            title, desc = (
                split_at(content, ":") if ":" in content else split_at(content, "-")
            )
        else:
            title = content
            desc = ""
        items.append(
            {
                "id": id,
                "title": title.strip(),
                "desc": desc.strip(),
            }
        )
    return items


class SubjectCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + "Based on that, enumerate a list of high-level subjects that I should learn to reach that GOAL based on the REASON. With each subject, write 3-6 sentences why each is important. Each subject should take less than 2 weeks each to get a general understanding of. These should be relatively high-level. Please say a couple sentences why each is important. Later, I will ask you for topics related to each subject, and then sub-topics and tasks related to those topics."
            " Return the result as a numbered list like the following:\n"
            "1. First subject\n"
            "2. Second subject\n"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TopicCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + 'Enumerate a list of topics for subject #{subject_id} "{subject_title}" with 3-6 sentences explaining why each is important.'
            " These topics should be in service of the specified GOAL and REASON. Each topic should take less than 3 days to get an understanding of."
            " Return the result as a numbered list like the following:\n"
            "1. First topic\n"
            "2. Second topic"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "subject_id",
                "subject_title",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class SubtopicCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + 'Enumerate a list of sub-topics that I should do to understand the topic #{topic_id} "{topic_title}" with 3-5 sentences explaining why it\'s important.'
            " The sub-topics should be in service of the specified GOAL and REASON. Each sub-topic should take less than 1 day to understand (for example, it shouldn't be to read an entire book), they shouldn't be too broad. Return the result as a numbered list like the following:"
            " Return the result as a numbered list like the following:\n"
            "1. First\n"
            "2. Second"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "topic_id",
                "topic_title",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + 'Enumerate a list of 1-4 tasks (no more than 4) that I should do to understand the sub-topic #{subtopic_id} "{subtopic_title}".'
            " The tasks should be in service of the specified GOAL and REASON. Each task should take less than 4 hours to understand (for example, it shouldn't be to read an entire book), they shouldn't be too broad. Return the result as a numbered list like the following:"
            " Return the result as a numbered list like the following:\n"
            "1. First\n"
            "2. Second"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "subtopic_id",
                "subtopic_title",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def get_subjects(
    chain: SubjectCreationChain,
    context: str,
) -> List[Dict]:
    response = chain.run(context=context)
    return parse_items(response)


async def get_topics(
    chain: TopicCreationChain,
    context: str,
    subject_id: int,
    subject_title: str,
) -> List[Dict]:
    response = await chain.arun(
        context=context,
        subject_id=subject_id,
        subject_title=subject_title,
    )
    return parse_items(response)


async def get_subtopics(
    chain: SubtopicCreationChain,
    context: str,
    topic_id: int,
    topic_title: str,
) -> List[Dict]:
    response = await chain.arun(
        context=context,
        topic_id=topic_id,
        topic_title=topic_title,
    )
    return parse_items(response)


async def get_tasks(
    chain: TaskCreationChain,
    context: str,
    subtopic_id: int,
    subtopic_title: str,
) -> List[Dict]:
    response = await chain.arun(
        context=context,
        subtopic_id=subtopic_id,
        subtopic_title=subtopic_title,
    )
    return parse_items(response, split_desc=False)


class ProfessorGPT(Chain, BaseModel):
    subject_creation_chain: SubjectCreationChain = Field(...)
    topic_creation_chain: TopicCreationChain = Field(...)
    subtopic_creation_chain: SubtopicCreationChain = Field(...)
    task_creation_chain: TaskCreationChain = Field(...)
    task_id_counter: int = Field(1)

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def build(model=MODEL, verbose=VERBOSE) -> "ProfessorGPT":
        llm = OpenAI(temperature=TEMPERATURE, model_name=model)
        return ProfessorGPT.from_llm(llm=llm, verbose=verbose)

    async def gather_tasks(self, context: str, subtopics: List[Dict]):
        jobs = []
        for subtopic in subtopics:
            if "desc" in subtopic and subtopic["desc"]:
                job = asyncio.create_task(
                    get_tasks(
                        chain=self.task_creation_chain,
                        context=context,
                        subtopic_id=subtopic["id"],
                        subtopic_title=subtopic["title"],
                    )
                )
                jobs.append(job)
        tasks = await asyncio.gather(*jobs)
        for subtopic, subtopic_tasks in zip(subtopics, tasks):
            subtopic["tasks"] = subtopic_tasks

    async def gather_subtopics(self, context: str, topics: List[Dict]):
        jobs = []
        for topic in topics:
            job = asyncio.create_task(
                get_subtopics(
                    chain=self.subtopic_creation_chain,
                    context=context,
                    topic_id=topic["id"],
                    topic_title=topic["title"],
                )
            )
            jobs.append(job)
        subtopics = await asyncio.gather(*jobs)

        jobs = []
        for topic, topic_subtopics in zip(topics, subtopics):
            topic["subtopics"] = topic_subtopics
            tc = context + subtopics_context(
                topic["id"], topic["title"], topic_subtopics
            )
            jobs.append(
                asyncio.create_task(
                    self.gather_tasks(
                        context=tc,
                        subtopics=topic_subtopics,
                    )
                )
            )
        await asyncio.gather(*jobs)

    async def gather_topics(self, context: str, subjects: List[Dict]):
        sc = context + subjects_context(subjects)
        jobs = []
        for subject in subjects:
            job = asyncio.create_task(
                get_topics(
                    chain=self.topic_creation_chain,
                    context=sc,
                    subject_id=subject["id"],
                    subject_title=subject["title"],
                )
            )
            jobs.append(job)
        topics = await asyncio.gather(*jobs)

        jobs = []
        for subject, subject_topics in zip(subjects, topics):
            subject["topics"] = subject_topics
            tc = context + topics_context(
                subject["id"], subject["title"], subject_topics
            )
            jobs.append(
                asyncio.create_task(
                    self.gather_subtopics(
                        context=tc,
                        topics=subject_topics,
                    )
                )
            )
        await asyncio.gather(*jobs)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        goal = inputs["goal"]
        reason = inputs["reason"]
        knowledge = inputs["knowledge"]
        pc = prelude_context(
            goal=inputs["goal"],
            reason=inputs["reason"],
            knowledge=inputs["knowledge"],
        )
        subjects = get_subjects(chain=self.subject_creation_chain, context=pc)
        asyncio.run(self.gather_topics(context=pc, subjects=subjects))

        md = gen_markdown(subjects, goal=goal, reason=reason, knowledge=knowledge)
        return {"markdown": md}

    @property
    def input_keys(self) -> List[str]:
        return ["goal", "reason", "knowledge"]

    @property
    def output_keys(self) -> List[str]:
        return ["markdown"]

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "ProfessorGPT":
        subject_creation_chain = SubjectCreationChain.from_llm(llm=llm, verbose=verbose)
        topic_creation_chain = TopicCreationChain.from_llm(llm=llm, verbose=verbose)
        subtopic_creation_chain = SubtopicCreationChain.from_llm(
            llm=llm, verbose=verbose
        )
        task_creation_chain = TaskCreationChain.from_llm(llm=llm, verbose=verbose)
        return cls(
            subject_creation_chain=subject_creation_chain,
            topic_creation_chain=topic_creation_chain,
            subtopic_creation_chain=subtopic_creation_chain,
            task_creation_chain=task_creation_chain,
            **kwargs,
        )

    def next_task_id(self):
        this_id = self.task_id_counter
        self.task_id_counter += 1
        return this_id


def run(goal: str, reason: str, knowledge: str):
    llm = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL)
    professor_gpt = ProfessorGPT.from_llm(llm=llm, verbose=VERBOSE)
    out = professor_gpt(
        {
            "goal": goal,
            "reason": reason,
            "knowledge": knowledge,
        }
    )
    return out.get("markdown", None)


if __name__ == "__main__":
    md = run(goal=GOAL, reason=REASON, knowledge=KNOWLEDGE)
    if md:
        with open(f"./examples/{TOPIC}_plan.md", "w") as f:
            f.write(md)
