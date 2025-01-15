
# Code Original: https://github.com/lorenzejay/agentic-rag-practical-example/tree/main

import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from agentic_rag.tools.weaviate_tool import WeaviateTool
from crewai_tools import EXASearchTool, WebsiteSearchTool
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# manager_llm = LLM(model="cerebras/llama-3.3-70b")
llm = LLM(model="cerebras/llama-3.3-70b")

manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
    llm=llm,
)


@CrewBase
class AgenticRagCrew:
    """AgenticRag crew"""

    @agent
    def document_rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["document_rag_agent"],
            tools=[WeaviateTool()],
            verbose=True,
            llm=llm,
        )

    @agent
    def web_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["web_agent"],
            tools=[EXASearchTool(), WebsiteSearchTool(config=dict(
        embedder=dict(provider="google", config=dict(model="models/text-embedding-004"))),
        llm=dict(provider="google",config=dict( model="gemini-1.5-pro")))],
            verbose=True,
            llm=llm,
        )

    @agent
    def code_execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["code_execution_agent"],
            verbose=True,
            llm=llm,
        )

    @task
    def fetch_tax_docs_task(self) -> Task:
        return Task(
            config=self.tasks_config["fetch_tax_docs_task"],
        )

    @task
    def answer_question_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_question_task"], output_file="report.md"
        )

    @task
    def business_trends_task(self) -> Task:
        return Task(config=self.tasks_config["business_trends_task"])

    @task
    def graph_visualization_task(self) -> Task:
        return Task(config=self.tasks_config["graph_visualization_task"])

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticRag crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=True,
            embedder={"provider": "google", "config": {"api_key": GOOGLE_API_KEY,"model": "models/text-embedding-004"}}

        )
