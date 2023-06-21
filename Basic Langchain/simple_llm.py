"""
The LLMs

The fundamental component of LangChain involves invoking an LLM with a specific input. To illustrate this, we'll explore a simple example. Let's imagine we are building a service that suggests personalized workout routines based on an individual's fitness goals and preferences.

To accomplish this, we will first need to import the LLM wrapper.

"""
from langchain.llms import OpenAI

"""The temperature parameter in OpenAI models manages the randomness of the output. When set to 0, the output is mostly predetermined and suitable for tasks requiring stability and the most probable result. At a setting of 1.0, the output can be inconsistent and interesting but isn't generally advised for most tasks. For creative tasks, a temperature between 0.70 and 0.90 offers a balance of reliability and creativity. The best setting should be determined by experimenting with different values for each specific use case. The code initializes the GPT-3 model’s Davinci variant."""

llm = OpenAI(model="text-davinci-003", temperature=0.9)

"""Now we can call it on some input!"""

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

print(llm(text))