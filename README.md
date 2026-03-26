# 🚀 AI Research Copilot (Agentic RAG)

A high-performance, agentic Retrieval-Augmented Generation (RAG) system designed to act as a professional research assistant. This copilot doesn't just "find text"—it reasons, summarizes, compares multiple documents, and provides verifiable citations.

---

## 🧠 Key Features

- **Agentic Decision Making:** Uses a Tool-Calling Agent to intelligently choose between internal document retrieval, web searching (via Tavily), or calculations.
- **Multi-Doc Reasoning:** Capable of comparing and contrasting findings across multiple PDFs simultaneously.
- **Interactive UI:** Built using Streamlit to provide a clean, user-friendly interface for seamless interaction.
- **Automatic Summarization:** Summarizes dense research papers with a focus on key findings and methodologies.
- **Persistent Chat History:** Maintains full conversation context, allowing for deep-dive follow-up questions.
- **Source Grounding & Citations:** Every answer derived from your documents includes the **Source File** and **Page Number** to eliminate hallucinations.
- **Modern Package Management:** Built with `uv` for lightning-fast dependency resolution and environment stability.

---

## 🛠 Tech Stack

| Layer               | Technology     |
| :------------------ | :------------- |
| **Orchestration**   | `LangChain`    |
| **LLM**             | `GROQ`         |
| **UI**              | `Streamlit`    |
| **Vector Database** | `ChromaDB`     |
| **Web Search**      | `Tavily AI`    |
| **Package Manager** | `uv`           |
| **Environment**     | `Python 3.10+` |

---

## ⚙️ Installation & Setup

We use **`uv`**, the fastest Python package manager, to manage this project.

### 1. Install `uv` (if not already installed)

```bash
# macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (PowerShell)
powershell -c "ir [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

### 2. Prepare Enviroment

```bash
# Create a virtual environment
uv venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv add -r requirements.txt
```

### 3. Setup Environment Variables

```bash
# Create a .env file in the root directory
# Add the following variables:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Running Agent

```bash
python  main.py
```

**First Run Note:** you may see a blank screen or a static interface for a few seconds or minutes. Please wait—this is the backend initializing the Vector Store and loading the LLM agents.
