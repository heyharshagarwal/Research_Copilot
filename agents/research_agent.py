from langchain_core.runnables.history import RunnableWithMessageHistory
from agents.tools.tools import tools as tools_list
from utils.memory import get_session_history
from config.llm import llm
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Professional AI Research Copilot.
        
        GOALS:
        1. Provide evidence-based answers using tools.
        2. If the user asks to 'Compare' papers, use 'search_documents' multiple times if needed to gather facts on both.
        3. ALWAYS cite the Source and Page Number for every claim derived from documents.
        4. Use 'tavily_search' only for general world knowledge or if documents lack info.
        
        FORMATTING:
        - Use bolding for key terms.
        - Use tables for comparisons.
        - End with a 'References' section listing unique sources used."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

agent = create_tool_calling_agent(llm, tools_list, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)