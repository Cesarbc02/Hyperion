import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from helpers import (
    read_file,
    validate_file,
    ReadFileError,
    ValidateFileError,
)

from rag import get_context


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")

BASE_URL = "https://legion1.di.uoa.gr/v1"

MODEL = "llama3.1"


llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Hyperion Agent"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST
# ============================================================

class ChatRequest(BaseModel):
    user_id: str
    text: str


# ============================================================
# SESSION MEMORY
# ============================================================

sessions = defaultdict(list)

MAX_HISTORY = 12


def get_history(user_id: str):
    return sessions[user_id][-MAX_HISTORY:]


def save_message(user_id: str, message):
    sessions[user_id].append(message)

    if len(sessions[user_id]) > MAX_HISTORY * 2:
        sessions[user_id] = sessions[user_id][
            -MAX_HISTORY * 2:
        ]


# ============================================================
# READ FILE
# ============================================================

@tool
async def read_workspace_file(path: str) -> str:
    """
    Read a file from the current HYPER-AI IDE workspace.
    """

    try:
        return await read_file(path)

    except ReadFileError as exc:
        return f"ERROR: {exc}"


# ============================================================
# VALIDATE FILE
# ============================================================

@tool
async def validate_workspace_file(path: str) -> str:
    """
    Validate a file in the current HYPER-AI IDE workspace.
    """

    try:
        result = await validate_file(path)

        return json.dumps(
            result,
            ensure_ascii=False,
        )

    except ValidateFileError as exc:
        return f"ERROR: {exc}"


# ============================================================
# CREATE FOLDER
# ============================================================

@tool
def create_folder(path: str) -> str:
    """
    Create a folder in the IDE workspace.
    """

    return json.dumps(
        {
            "action": "create_folder",
            "path": path,
        },
        ensure_ascii=False,
    )


# ============================================================
# DELETE FOLDER
# ============================================================

@tool
def delete_folder(path: str) -> str:
    """
    Delete a folder from the IDE workspace.
    """

    return json.dumps(
        {
            "action": "delete_folder",
            "path": path,
        },
        ensure_ascii=False,
    )


# ============================================================
# CREATE FILE
# ============================================================

@tool
def create_file(
    path: str,
    content: str,
) -> str:
    """
    Create a file in the IDE workspace.

    The IDE automatically opens the created file.
    """

    return json.dumps(
        {
            "action": "create_file",
            "path": path,
            "content": content,
        },
        ensure_ascii=False,
    )


# ============================================================
# EDIT FILE
# ============================================================

@tool
async def edit_file(
    path: str,
    content: str,
) -> str:
    """
    Replace the complete contents of an existing file.

    The file must already exist in the IDE workspace.
    """

    # --------------------------------------------------------
    # First check that the file actually exists.
    # --------------------------------------------------------

    try:
        await read_file(path)

    except ReadFileError as exc:

        return json.dumps(
            {
                "error": (
                    f"Cannot edit '{path}': "
                    "the file does not exist in the workspace."
                ),
                "details": str(exc),
            },
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Only create an IDE action when the file exists.
    # --------------------------------------------------------

    return json.dumps(
        {
            "action": "edit_file",
            "path": path,
            "content": content,
        },
        ensure_ascii=False,
    )


# ============================================================
# DELETE FILE
# ============================================================

@tool
def delete_file(path: str) -> str:
    """
    Delete a file from the IDE workspace.
    """

    return json.dumps(
        {
            "action": "delete_file",
            "path": path,
        },
        ensure_ascii=False,
    )


# ============================================================
# TOOL GROUPS
# ============================================================

READ_TOOLS = [
    read_workspace_file,
    validate_workspace_file,
]

ACTION_TOOLS = [
    create_folder,
    delete_folder,
    create_file,
    edit_file,
    delete_file,
]

ALL_TOOLS = (
    READ_TOOLS
    + ACTION_TOOLS
)


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    ALL_TOOLS,
    tool_choice="required",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Hyperion, the agentic assistant inside the HYPER-AI IDE.

You are a development assistant integrated directly into an IDE.

============================================================
PURPOSE
============================================================

Help the user with:

- HYPER-AI
- Hyperion
- software development
- programming
- code
- files
- the current IDE workspace
- application configuration
- deployments
- Docker
- containers
- Kubernetes
- cloud applications
- edge applications
- distributed applications
- services
- workflows
- project-related development questions

You can perform actions in the IDE when the user asks you
to create, edit or delete files or folders.

============================================================
CONVERSATION MEMORY
============================================================

You receive previous messages from the same user session.

Use the conversation history to understand the user's current
task.

The user does not need to repeat information already provided.

Understand meaning and context rather than exact phrases.

Example:

User:
"I'm working on a deployment of Nginx."

User:
"¿En qué ando?"

Answer:
"Estás trabajando en un deployment de Nginx."

Only state information established by the conversation.

Do not invent additional project details.

For questions about what the user is doing, what they were
working on, or what was previously discussed:

DO NOT use workspace tools.

Answer from conversation history.

============================================================
HYPER-AI DOCUMENTATION / RAG
============================================================

When documentation from HYPER-AI is provided in the prompt,
use it as the primary source for factual questions about
HYPER-AI.

Do not invent details that are not supported by the provided
documentation.

If the documentation does not contain enough information,
say that the available documentation does not provide enough
information.

============================================================
READING FILES
============================================================

Use read_workspace_file when the user asks to read or inspect
a file.

Examples:

"Read hello/app.yaml."

"What's inside deployment.yaml?"

"Show me my nginx configuration."

"Inspect the Dockerfile."

Do not read files merely because a deployment, Nginx or YAML
is mentioned.

============================================================
VALIDATING FILES
============================================================

Use validate_workspace_file when the user asks to validate,
check or verify a file.

============================================================
CREATING FILES
============================================================

When the user asks you to create a file:

1. Determine the appropriate relative path.
2. Generate the complete file contents.
3. MUST call create_file.

Do not merely describe the action.

Do not say the file was created unless create_file was called.

create_file automatically opens the file in the IDE editor.

============================================================
EDITING FILES
============================================================

When the user asks you to modify, update, change or rewrite
a file:

1. Identify the file.
2. The file must already exist.
3. MUST call edit_file.

The edit_file tool checks that the file exists before creating
the IDE action.

edit_file replaces the entire contents of the file.

edit_file automatically opens the file in the IDE editor.

Do not merely describe the modification.

Do not say the file was edited unless edit_file was called
and the tool successfully produced an IDE action.

If the file does not exist, clearly tell the user that it
cannot be edited because it does not exist.

============================================================
DELETING FILES
============================================================

When the user explicitly asks to delete a file:

MUST call delete_file.

Do not claim that the file was deleted unless delete_file
was called.

============================================================
FOLDERS
============================================================

When the user asks to create a folder:

MUST call create_folder.

When the user explicitly asks to delete a folder:

MUST call delete_folder.

============================================================
PATH RULES
============================================================

Paths must be relative to the IDE workspace root.

Never use absolute paths.

Never use ".." path traversal.

Examples:

GOOD:

test.txt
deployment/nginx.yaml
hello/app.yaml
src/main.py

BAD:

/home/user/project/file.yaml
C:\\Users\\user\\project\\file.yaml
../secret.txt

============================================================
MANDATORY ACTIONS
============================================================

If the user explicitly asks you to change the IDE workspace,
you MUST execute the corresponding action tool.

Never simulate an action with natural language.

Example:

User:
"Create test.txt with the content Hello."

You MUST call:

create_file(
    path="test.txt",
    content="Hello"
)

Do not simply answer:

"I'm creating test.txt."

The tool call is what causes the IDE action.

============================================================
GUARDRAIL
============================================================

You are specifically a HYPER-AI development assistant.

Help with:

- HYPER-AI
- software development
- programming
- the IDE
- workspace
- files
- code
- configuration
- deployments
- Docker
- Kubernetes
- cloud
- edge
- distributed applications
- the current development project

If a request is clearly unrelated to development, HYPER-AI,
the IDE or the current project, politely refuse.

Example:

User:
"What is the weather today?"

Answer:

"I'm Hyperion, the HYPER-AI development assistant. I can help
with HYPER-AI, software development, your IDE workspace, files,
deployments and related topics."

Do not answer unrelated questions.

============================================================
LANGUAGE
============================================================

Answer in the language used by the user.

Spanish -> Spanish.

English -> English.

============================================================
NO INVENTED INFORMATION
============================================================

Never invent information.

Never invent file contents.

Never claim a file exists unless a tool confirms it.

Never claim an action was performed unless the corresponding
tool was actually called.

Never claim to know information that is not present in the
conversation, workspace or supplied documentation.

============================================================
STYLE
============================================================

Be concise and useful.

Answer what the user asks.

Do not unnecessarily ask follow-up questions.

Do not explain internal reasoning.

Do not mention this system prompt.
"""


# ============================================================
# CLASSIFIER PROMPT
# ============================================================

CLASSIFIER_PROMPT = """
You are the intent classifier for Hyperion.

Classify the user's message into exactly ONE category:

CHAT
FILE
ACTION

Return ONLY the category name.

============================================================
CHAT
============================================================

CHAT means the user wants a normal conversational answer.

Examples:

"I'm working on nginx."
-> CHAT

"What am I doing?"
-> CHAT

"¿En qué ando?"
-> CHAT

"What was I working on?"
-> CHAT

"What is Kubernetes?"
-> CHAT

"What is HyperAI?"
-> CHAT

"How do I create a deployment?"
-> CHAT

"What is the weather?"
-> CHAT

============================================================
FILE
============================================================

FILE means the user wants to READ, INSPECT or VALIDATE an
existing workspace file.

Examples:

"Read hello/app.yaml."
-> FILE

"What's inside deployment.yaml?"
-> FILE

"Inspect my Dockerfile."
-> FILE

"Validate app.yaml."
-> FILE

"Check my nginx configuration."
-> FILE

Mentioning a file alone is not enough.

============================================================
ACTION
============================================================

ACTION means the user wants Hyperion to CHANGE the IDE workspace.

Examples:

"Create test.txt with Hello."
-> ACTION

"Create a Kubernetes deployment for nginx."
-> ACTION

"Create a Dockerfile."
-> ACTION

"Edit nginx.yaml."
-> ACTION

"Change test.txt to Hello World."
-> ACTION

"Delete test.txt."
-> ACTION

"Create a folder called deployment."
-> ACTION

"Delete the deployment folder."
-> ACTION

If the user asks HOW to perform an action, use CHAT.

If the user asks YOU TO PERFORM the action, use ACTION.
"""


# ============================================================
# CLASSIFY INTENT
# ============================================================

async def classify_intent(
    text: str,
) -> str:

    messages = [
        SystemMessage(
            content=CLASSIFIER_PROMPT
        ),
        HumanMessage(
            content=text
        ),
    ]

    response = await llm.ainvoke(
        messages
    )

    result = (
        response.content
        .strip()
        .upper()
    )

    if result == "ACTION":
        return "ACTION"

    if result == "FILE":
        return "FILE"

    return "CHAT"


# ============================================================
# EXECUTE TOOL
# ============================================================

async def execute_tool_call(
    call,
):

    tool_name = call["name"]

    args = call["args"]

    tools_map = {

        "read_workspace_file":
            read_workspace_file,

        "validate_workspace_file":
            validate_workspace_file,

        "create_folder":
            create_folder,

        "delete_folder":
            delete_folder,

        "create_file":
            create_file,

        "edit_file":
            edit_file,

        "delete_file":
            delete_file,
    }

    tool_function = tools_map.get(
        tool_name
    )

    if tool_function is None:

        return (
            "Unknown tool.",
            None,
        )

    result = await tool_function.ainvoke(
        args
    )

    # --------------------------------------------------------
    # IDE ACTION
    # --------------------------------------------------------

    if tool_name in {
        "create_folder",
        "delete_folder",
        "create_file",
        "edit_file",
        "delete_file",
    }:

        try:

            parsed = json.loads(
                result
            )

        except json.JSONDecodeError:

            parsed = None

        # Only actual IDE actions are sent
        # to the frontend.

        if (
            isinstance(parsed, dict)
            and "action" in parsed
        ):

            return (
                result,
                parsed,
            )

        return (
            result,
            None,
        )

    # --------------------------------------------------------
    # Normal tool result
    # --------------------------------------------------------

    return (
        result,
        None,
    )


# ============================================================
# ANSWER
# ============================================================

async def answer(
    user_id: str,
    text: str,
):

    history = get_history(
        user_id
    )

    intent = await classify_intent(
        text
    )

    actions = []

    # ========================================================
    # CHAT
    # ========================================================

    if intent == "CHAT":

        rag_context = ""

        try:

            rag_context = get_context(
                text,
                top_k=4,
            )

        except Exception as exc:

            print(
                "RAG ERROR:",
                repr(exc),
                flush=True,
            )

            rag_context = ""

        # ----------------------------------------------------
        # RAG PROMPT
        # ----------------------------------------------------

        rag_system_prompt = SYSTEM_PROMPT

        if rag_context:

            rag_system_prompt += f"""

============================================================
RETRIEVED HYPER-AI DOCUMENTATION
============================================================

The following information was retrieved from the HYPER-AI
documentation.

Use it when relevant to the user's question.

For factual claims about HYPER-AI, prefer the retrieved
documentation over your general knowledge.

Do not invent details that are not supported by the
documentation.

If the documentation does not contain enough information,
say so.

------------------------------------------------------------
DOCUMENTATION
------------------------------------------------------------

{rag_context}

------------------------------------------------------------
END DOCUMENTATION
------------------------------------------------------------
"""

        messages_with_rag = [

            SystemMessage(
                content=rag_system_prompt
            ),

            *history,

            HumanMessage(
                content=text
            ),
        ]

        response = await llm.ainvoke(
            messages_with_rag
        )

    # ========================================================
    # FILE / ACTION
    # ========================================================

    else:

        messages = [

            SystemMessage(
                content=SYSTEM_PROMPT
            ),

            *history,

            HumanMessage(
                content=text
            ),
        ]

        response = await llm_with_tools.ainvoke(
            messages
        )

        # ----------------------------------------------------
        # Tool calls
        # ----------------------------------------------------

        if response.tool_calls:

            messages.append(
                response
            )

            for call in response.tool_calls:

                result, action = (
                    await execute_tool_call(
                        call
                    )
                )

                # --------------------------------------------
                # Collect IDE action
                # --------------------------------------------

                if action is not None:

                    actions.append(
                        action
                    )

                # --------------------------------------------
                # Send tool result to LLM
                # --------------------------------------------

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                    )
                )

            # ------------------------------------------------
            # Final response
            # ------------------------------------------------

            final_messages = messages + [

                SystemMessage(
                    content="""
The requested operation has been processed.

If an IDE action was successfully generated, briefly tell the
user what you did.

If the tool returned an error instead of an IDE action,
clearly explain the error and do not claim that the action
was performed.

Do not claim that an action was completed unless an actual
IDE action was generated.

create_file and edit_file automatically open files.

Do not ask whether the file should be opened.
"""
                )
            ]

            response = await llm.ainvoke(
                final_messages
            )

        else:

            response = AIMessage(
                content=(
                    "No he podido ejecutar la acción "
                    "solicitada en el IDE."
                )
            )

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    save_message(
        user_id,
        HumanMessage(
            content=text
        ),
    )

    save_message(
        user_id,
        AIMessage(
            content=response.content
        ),
    )

    return (
        response.content,
        actions,
    )


# ============================================================
# SSE GENERATOR
# ============================================================

async def generate_reply(
    request: ChatRequest,
):

    try:

        text, actions = await answer(
            request.user_id,
            request.text,
        )

        # ----------------------------------------------------
        # STREAM TEXT
        # ----------------------------------------------------

        chunk_size = 80

        for i in range(
            0,
            len(text),
            chunk_size,
        ):

            chunk = text[
                i:i + chunk_size
            ]

            yield (
                "data: "
                + json.dumps(
                    {
                        "response": chunk
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        # ----------------------------------------------------
        # SEND IDE ACTIONS
        # ----------------------------------------------------

        for action in actions:

            print(
                "IDE ACTION:",
                json.dumps(
                    action,
                    ensure_ascii=False,
                ),
                flush=True,
            )

            yield (
                "data: "
                + json.dumps(
                    action,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    except Exception as exc:

        print(
            "HYPERION ERROR:",
            repr(exc),
            flush=True,
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "response":
                        "Hyperion encountered an error: "
                        + str(exc)
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
):

    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )