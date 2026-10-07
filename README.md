# Hyperion — HYPER-AI Agentic Assistant

Hyperion is an LLM-powered agentic assistant for the HYPER-AI IDE.

It allows users to interact with the IDE using natural language and can answer questions, retrieve information from the HYPER-AI documentation, maintain session memory, inspect files, validate files, and perform workspace actions such as creating, editing, and deleting files and folders.

## Features

- LLM-powered natural-language interaction
- Session memory using `user_id`
- RAG over the HYPER-AI documentation
- FAISS-based vector search
- Workspace file reading
- File validation
- Create files and folders
- Edit existing files
- Delete files and folders
- SSE streaming responses
- Dockerized deployment
- IDE backend integration
- Guardrail for unsupported requests such as weather queries

## Architecture

```text
HYPER-AI IDE
     |
     | POST /chat
     v
+----------------------+
|      Hyperion        |
|                      |
|  FastAPI / SSE       |
|  Session Memory      |
|  Intent Classifier   |
|  LLM + Tools         |
|  RAG / FAISS         |
+----------+-----------+
           |
     +-----+------+
     |            |
     v            v
  LLM Server   IDE Backend
                 |
                 v
              HYPER-AI
                 IDE