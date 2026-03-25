from agents.research_agent import agent_with_chat_history

def chat():
    print("🚀 AI Research Copilot is ready! (Type 'exit' to stop)")
    
    # In a real app, this ID would be unique to each user/tab
    session_id = "test_user_123" 
    config = {"configurable": {"session_id": session_id}}

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # This calls the agent, which calls the tool, which searches ChromaDB
        response = agent_with_chat_history.invoke(
            {"input": user_input},
            config=config
        )

        print(f"\n🤖 Agent: {response['output']}")

if __name__ == "__main__":
    chat()