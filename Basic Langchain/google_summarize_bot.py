from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0)

search = GoogleSearchAPIWrapper()

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)


summarize_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

response = agent("What's the latest news about the Mars rover?")
print(response['output'])