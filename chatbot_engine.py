import os
from typing import List

import langchain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.tools import BaseTool

langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    persist_dir = "store-langchain-docs"
    if os.path.exists(persist_dir):
        print('Import persistent data...')
        vectorstore = Chroma(
            embedding_function=VectorstoreIndexCreator().embedding,
            persist_directory=persist_dir
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        print('Loading documents...')
        loader = DirectoryLoader("./docs/", glob="**/*.html",
                                 loader_kwargs={"encoding": "utf-8"},
                                 show_progress=True)
        return VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            vectorstore_kwargs={"persist_directory": persist_dir},
        ).from_loaders([loader])

def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name="LangChain Documentation Index",
        description="A comprehensive index of LangChain documentation, including guides, reference materials, and examples, intended to assist developers in integrating and utilizing LangChain's capabilities in their projects."
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()


def chat(
    message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper
) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = create_tools(index, llm)

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory
    )

    return agent_chain.run(input=message)
