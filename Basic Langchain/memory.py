"""### The Memory

In LangChain, Memory refers to the mechanism that stores and manages the conversation history between a user and the AI. It helps maintain context and coherency throughout the interaction, enabling the AI to generate more relevant and accurate responses. Memory, such as `ConversationBufferMemory`, acts as a wrapper around `ChatMessageHistory`, extracting the messages and providing them to the chain for better context-aware generation."""

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)

#Output

"""**> Entering new ConversationChain chain...**

Prompt after formatting:

***The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Tell me about yourself.
AI:  Hi there! I'm an AI created to help people with their daily tasks. I'm programmed to understand natural language and respond to questions and commands. I'm also able to learn from my interactions with people, so I'm constantly growing and improving. I'm excited to help you out!
Human: What can you do?
AI:  I can help you with a variety of tasks, such as scheduling appointments, setting reminders, and providing information. I'm also able to answer questions about topics like current events, sports, and entertainment. I'm always learning new things, so I'm sure I can help you with whatever you need.
Human: How can you help me with data analysis?
AI:*  I'm not familiar with data analysis, but I'm sure I can help you find the information you need. I can search the web for articles and resources related to data analysis, and I can also provide you with links to helpful websites.**

**> Finished chain.**

"""