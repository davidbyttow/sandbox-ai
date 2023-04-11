import os
from collections import deque
from typing import Dict, List, Optional, Any

import faiss
from pydantic import BaseModel, Field
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.chains.base import Chain
from langchain.docstore import InMemoryDocstore


TEMPERATURE = 0
EMBEDDING_SIZE = 1536
VERBOSE = True

embeddings_model = OpenAIEmbeddings()
index = faiss.IndexFlatL2(EMBEDDING_SIZE)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


class SubjectCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + "Based on that, enumerate a list of high-level subjects that are worth learning to reach that goal based on the reason. Each subject should take less than 2 weeks each to get a general understanding of. These should be relatively high-level. Please say a couple sentences why each is important. Later, I will ask you for topics related to each subject, and then sub-topics or lessons or exercises related to those topics."
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
            + 'Enumerate a list of topics for subject #{subject_id} "{subject_title}" with a few sentences explaining why each is important.'
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


class LessonCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        template = (
            "{context}"
            + 'Enumerate a list of sub-topics for topic #{topic_id} "{topic_title}" with a few sentences explaining why each is important.'
            " These topics should be in service of the specified GOAL and REASON. Each topic/lesson should take less than 1 day to complete and understand, they shouldn't be too broad."
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


def prelude_context(goal: str, reason: str, prior_knowledge: str):
    return (
        "You are a professor and tutor. You are generating a learning plan and creating lessons/exercises for your private student. Your student wants to learn something effectively and has a fixed amount of time they'd like to do so. They have may prior knowledge already so you should take that into account.\n"
        f'Here\'s what the student wants to learn (GOAL): "{goal}"\n'
        f'Here\'s why they want to learn it (REASON): "{reason}"\n'
        f'Here\'s what they already know related to the topic (KNOWLEDGE): "{prior_knowledge}"\n\n'
    )


def enumerate_items(items: List[Dict]):
    return "\n".join([f"{s['id']}. {s['title']}" for s in items]) + "\n"


def subjects_context(subjects: List[Dict]):
    return (
        "Here are the subjects you told them to focus on:\n"
        + enumerate_items(subjects)
        + "\n\n"
    )


def topics_context(subject_title: str, topics: List[Dict]):
    return (
        f'Based on the subject "{subject_title}", here are the topics you came up with for it to learn:\n'
        + enumerate_items(topics)
        + "\n\n"
    )


def split_at(text: str, sep: str) -> tuple:
    index = text.find(sep)
    return (text[:index], text[index + 1 :]) if index != -1 else (text, "")


def parse_items(response: str) -> List[str]:
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
        title, desc = (
            split_at(content, ":") if ":" in content else split_at(content, "-")
        )
        items.append(
            {
                "id": id,
                "title": title.strip(),
                "desc": desc.strip(),
            }
        )
    return items


def get_subjects(
    chain: SubjectCreationChain,
    context: str,
) -> List[str]:
    response = chain.run(context=context)
    return parse_items(response)


def get_topics(
    chain: TopicCreationChain,
    context: str,
    subject_id: int,
    subject_title: str,
) -> List[Dict]:
    response = chain.run(
        context=context,
        subject_id=subject_id,
        subject_title=subject_title,
    )
    return parse_items(response)


def get_lessons(
    chain: LessonCreationChain,
    context: str,
    topic_id: int,
    topic_title: str,
) -> List[Dict]:
    response = chain.run(
        context=context,
        topic_id=topic_id,
        topic_title=topic_title,
    )
    print(response)
    return parse_items(response)


class ProfessorGPT(Chain, BaseModel):
    subject_creation_chain: SubjectCreationChain = Field(...)
    topic_creation_chain: TopicCreationChain = Field(...)
    lesson_creation_chain: LessonCreationChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)

    class Config:
        arbitrary_types_allowed = True

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pc = prelude_context(
            goal=inputs["goal"],
            reason=inputs["reason"],
            prior_knowledge=inputs["prior_knowledge"],
        )
        subjects = get_subjects(chain=self.subject_creation_chain, context=pc)
        sc = pc + subjects_context(subjects)
        for subject in subjects:
            topics = get_topics(
                chain=self.topic_creation_chain,
                context=sc,
                subject_id=subject["id"],
                subject_title=subject["title"],
            )
            subject["topics"] = topics
            tc = pc + topics_context(subject["title"], topics)
            for topic in topics:
                lessons = get_lessons(
                    chain=self.lesson_creation_chain,
                    context=tc,
                    topic_id=topic["id"],
                    topic_title=topic["title"],
                )
                topic["lessons"] = lessons
        print(subjects)
        return {}

    @property
    def input_keys(self) -> List[str]:
        return ["goal", "reason", "prior_knowledge"]

    @property
    def output_keys(self) -> List[str]:
        return []

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "ProfessorGPT":
        subject_creation_chain = SubjectCreationChain.from_llm(llm=llm, verbose=verbose)
        topic_creation_chain = TopicCreationChain.from_llm(llm=llm, verbose=verbose)
        lesson_creation_chain = LessonCreationChain.from_llm(llm=llm, verbose=verbose)
        return cls(
            subject_creation_chain=subject_creation_chain,
            topic_creation_chain=topic_creation_chain,
            lesson_creation_chain=lesson_creation_chain,
            vectorstore=vectorstore,
            **kwargs,
        )

    def next_task_id(self):
        this_id = self.task_id_counter
        self.task_id_counter += 1
        return this_id


llm = OpenAI(temperature=TEMPERATURE, model_name="gpt-3.5-turbo")

professor_gpt = ProfessorGPT.from_llm(llm=llm, vectorstore=vectorstore, verbose=VERBOSE)

professor_gpt(
    {
        "goal": "I want to learn how to code neural networks.",
        "reason": "I want to be able to code FCN, FNN, CNNs and GANs by hand using pytorch, but also understand the math.",
        "prior_knowledge": "I am an expert in python already and have a rough knowledge of linear algebra and statistics.",
    }
)
