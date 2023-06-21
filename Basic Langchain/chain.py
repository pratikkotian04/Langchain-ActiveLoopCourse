"""### The Chains

In LangChain, a chain is an end-to-end wrapper around multiple individual components, providing a way to accomplish a common use case by combining these components in a specific sequence. The most commonly used type of chain is the `LLMChain`, which consists of a `PromptTemplate`, a model (either an LLM or a ChatModel), and an optional output parser.

The `LLMChain` works as follows:

1. Takes (multiple) input variables.
2. Uses the `PromptTemplate` to format the input variables into a prompt.
3. Passes the formatted prompt to the model (LLM or ChatModel).
4. If an output parser is provided, it uses the `OutputParser` to parse the output of the LLM into a final format.

In the next example, we demonstrate how to create a chain that generates a possible name for a company that produces eco-friendly water bottles. By using LangChain's `LLMChain`, `PromptTemplate`, and `OpenAI`classes, we can easily define our prompt, set the input variables, and generate creative outputs"""

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("Eco-friendly Clothes"))