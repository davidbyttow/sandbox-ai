from pathlib import Path

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

dir = Path(__file__).parent

db_path = str(dir / "localdata/cryptobat_development.db")
db = SQLDatabase.from_uri(
    "postgresql://sorare:@localhost:5432/cryptobat_development",
)


llm = OpenAI(temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

output = agent_executor.run("top 10 NBA players with the highest tenGameAverage?")

print(output)
