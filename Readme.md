# 🦜 LangChain Notes

A structured set of notes covering LangChain fundamentals — from core concepts to building RAG pipelines and agents.

---

## Table of Contents

1. [Introduction to LangChain](#1-introduction-to-langchain)
2. [LangChain Components](#2-langchain-components)
3. [Models](#3-models)
4. [Prompts](#4-prompts)
5. [Structured Output](#5-structured-output)
6. [Output Parsers](#6-output-parsers)
7. [Chains](#7-chains)
8. [Runnables](#8-runnables)
9. [LCEL (LangChain Expression Language)](#9-lcel-langchain-expression-language)
10. [Document Loaders](#10-document-loaders)
11. [Text Splitters](#11-text-splitters)
12. [Vector Stores](#12-vector-stores)
13. [Retrievers](#13-retrievers)
14. [RAG (Retrieval-Augmented Generation)](#14-rag-retrieval-augmented-generation)
15. [RAG using LangChain](#15-rag-using-langchain)
16. [Tools](#16-tools)
17. [Tool Calling](#17-tool-calling)
18. [Agents](#18-agents)

---

## 1. Introduction to LangChain

### What is LangChain?
LangChain is an open-source framework for developing applications powered by large language models (LLMs). It provides modular components and end-to-end tools that help developers build complex AI applications such as chatbots, question-answering systems, retrieval-augmented generation (RAG) systems, autonomous agents, and more.

**Why LangChain:**
- Supports all the major LLMs
- Simplifies developing LLM-based applications
- Integrations available for all major tools
- Open source / free / actively developed
- Supports all major GenAI use cases

### Why do we need LangChain?
Example queries an LLM app built with LangChain can handle:
- Explain page number 5 as if I am a 5 year old
- Generate a True/False exercise on Linear Regression
- Generate notes for Decision Trees

**Benefits:**
- Concept of chains
- Model-agnostic development
- Complete ecosystem
- Memory and state handling

### What can you build?
- Conversational chatbots
- AI knowledge assistants
- AI agents
- Workflow automation
- Summarization / research helpers

### Alternatives
- LlamaIndex
- Haystack

---

## 2. LangChain Components

Core building blocks of the framework:
- **Models** – interfaces to interact with AI models
- **Prompts** – dynamic, reusable instructions for models
- **Chains** – sequences of calls linked together
- **Indexes** – connect the app to external knowledge sources
- **Memory** – persist state across LLM calls
- **Agents** – decide and act using tools

### Models
In LangChain, "models" are the core interfaces through which you interact with AI models.

### Prompts
1. Dynamic & reusable prompts
2. Role-based prompts
3. Few-shot prompting

### Chains
Chains link multiple steps/components together into a single pipeline.

### Indexes
Indexes connect your application to external knowledge — such as PDFs, websites, or databases.

### Memory
LLM API calls are stateless by default. Memory types:
- **ConversationBufferMemory** – stores a transcript of recent messages. Great for short chats but can grow large quickly.
- **ConversationBufferWindowMemory** – only keeps the last N interactions to avoid excessive token usage.
- **Summarizer-Based Memory** – periodically summarizes older chat segments to keep a condensed memory footprint.
- **Custom Memory** – for advanced use cases, store specialized state (e.g., user preferences or key facts) in a custom memory class.

### Agents
Systems that can decide which actions to take using tools.

---

## 3. Models

### What are Models?
The Model component in LangChain is a crucial part of the framework, designed to facilitate interactions with various language models and embedding models. It abstracts the complexity of working directly with different LLMs, chat models, and embedding models, providing a uniform interface to communicate with them. This makes it easier to build applications that rely on AI-generated text, text embeddings for similarity search, and retrieval-augmented generation (RAG).

### Language Models
Language Models are AI systems designed to process, generate, and understand natural language text.

- **LLMs** – General-purpose models used for raw text generation. They take a string (plain text) as input and return a string. These are traditionally older models and are not used much now.
- **Chat Models** – Language models specialized for conversational tasks. They take a sequence of messages as input and return chat messages as output. These are traditionally newer and more commonly used than LLMs.

### Demo Providers
1. OpenAI
2. Anthropic
3. Google
4. HuggingFace

### Open Source Models
Open-source language models are freely available AI models that can be downloaded, modified, fine-tuned, and deployed without restrictions from a central provider. Unlike closed-source models such as OpenAI's GPT-4, Anthropic's Claude, or Google's Gemini, open-source models allow full control and customization.

- **Where to find them:** HuggingFace — the largest repository of open-source LLMs.

### Embedding Models
Models used to convert text into vector representations for similarity search and retrieval.

---

## 4. Prompts

### What are Prompts?
Prompts are the input instructions or queries given to a model to guide its output.

### Static vs Dynamic Prompts
Example (dynamic prompt via a Streamlit UI):

```python
paper_input = st.selectbox("Select Research Paper Name", [
    "Select...",
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])

length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"
])
```

Prompt template used:

```
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with:
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
```

### Prompt Template
A `PromptTemplate` in LangChain is a structured way to create prompts dynamically by inserting variables into a predefined template. Instead of hardcoding prompts, `PromptTemplate` allows you to define placeholders that can be filled in at runtime with different inputs. This makes it reusable, flexible, and easy to manage — especially with dynamic user inputs or automated workflows.

**Why use `PromptTemplate` over f-strings?**
1. Default validation
2. Reusable
3. Fits the LangChain ecosystem

### Messages & Chat Prompt Templates
Structures for representing conversational turns (system / human / AI messages) that can be templated like regular prompts.

### Message Placeholder
A `MessagesPlaceholder` in LangChain is a special placeholder used inside a `ChatPromptTemplate` to dynamically insert chat history or a list of messages at runtime.

---

## 5. Structured Output

### What is Structured Output?
In LangChain, structured output refers to having language models return responses in a well-defined data format (e.g., JSON) rather than free-form text. This makes the model's output easier to parse and work with programmatically.

*Reference: "Structured Outputs from LLMs" — LangChain output parsers, via LinkedIn (Vijay Chaudhary).*

### Why do we need Structured Output?
- Data extraction
- API building
- Agents

### Ways to Get Structured Output

**`with_structured_output`**
Example system prompt:
```
You are an AI assistant that extracts structured insights from text.
Given a product review, extract:
- Summary: A brief overview of the main points.
- Sentiment: Overall tone of the review (positive, neutral, negative).
Return the response in JSON format.
```

**TypedDict**
`TypedDict` is a way to define a dictionary in Python where you specify what keys and values should exist. It helps ensure your dictionary follows a specific structure.

*Why use TypedDict?*
- Tells Python what keys are required and what types of values they should have.
- Does **not** validate data at runtime — it only helps with type hints for cleaner code.

Variants covered:
- Simple TypedDict
- Annotated TypedDict
- Literal
- More complex forms (with pros and cons)

**Pydantic**
Pydantic is a data validation and parsing library for Python. It ensures the data you work with is correct, structured, and type-safe.

Topics covered:
- Basic example
- Default values
- Optional fields
- Type coercion
- Built-in validation
- `Field` function → default values, constraints, description, regex expressions
- Returns a Pydantic object → convert to JSON/dict

**JSON Schema**
Defining output structure directly via JSON Schema.

### When to Use What?
Choosing between `TypedDict`, Pydantic, and JSON Schema depending on whether you need runtime validation, simple type hints, or schema portability.

---

## 6. Output Parsers

Output Parsers in LangChain help convert raw LLM responses into structured formats like JSON, CSV, Pydantic models, and more. They ensure consistency, validation, and ease of use in applications.

- **StrOutputParser** – the simplest output parser; parses the LLM output and returns it as a plain string.
- **JSONOutputParser** – parses model output into JSON.
- **StructuredOutputParser** – extracts structured JSON data from LLM responses based on predefined field schemas (`ResponseSchema`), ensuring the output follows a structured format.
- **PydanticOutputParser** – parses LLM output directly into a Pydantic model.

---

## 7. Chains

- **Simple Chain** – a single-step pipeline (prompt → model → parser).
- **Sequential Chain** – multiple steps executed one after another.
- **Parallel Chain** – multiple chains executed simultaneously on the same input.
- **Conditional Chain** – routes execution down different paths based on conditions.

---

## 8. Runnables

### The Why
Runnables provide a **standard, composable interface** so every LangChain component (models, prompts, parsers, retrievers) can be chained together predictably, invoked, batched, or streamed the same way.

### The What
A `Runnable` is the core abstraction in LangChain — any component that implements `.invoke()`, `.batch()`, and `.stream()` and can be composed with other runnables.

---

## 9. LCEL (LangChain Expression Language)

LCEL is the syntax (`|` pipe operator) used to compose Runnables into pipelines.

### Core Runnable Primitives

1. **RunnableSequence** — a sequential chain of runnables that executes each step one after another, passing the output of one step as the input to the next. Useful for composing multiple runnables into a structured workflow.

2. **RunnableParallel** — a runnable primitive that allows multiple runnables to execute in parallel. Each runnable receives the same input and processes it independently, producing a dictionary of outputs.

3. **RunnablePassthrough** — a special primitive that simply returns the input as output without modifying it.

4. **RunnableLambda** — allows you to apply custom Python functions within an AI pipeline. Acts as middleware between different AI components, enabling preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow.

5. **RunnableBranch** — a control-flow component that conditionally routes input data to different chains or runnables based on custom logic. Functions like an if/elif/else block for chains: you define condition functions each associated with a runnable (LLM call, prompt chain, tool, etc.). The first matching condition executes; if none match, a default runnable is used (if provided).

---

## 10. Document Loaders

Document loaders load data from various sources into a standardized format (usually `Document` objects), used later for chunking, embedding, retrieval, and generation.

- **TextLoader** – reads plain text (`.txt`) files into `Document` objects.
  - *Use case:* chat logs, scraped text, transcripts, code snippets, or any plain text data.
  - *Limitation:* works only with `.txt` files.

- **PyPDFLoader** – loads content from PDF files, converting each page into a `Document` object.
  - *Limitation:* uses the PyPDF library under the hood — not great with scanned PDFs or complex layouts.

- **DirectoryLoader** – loads multiple documents from a directory (folder) of files.

- **Load vs. Lazy Load** – `.load()` loads everything into memory at once; lazy loading streams documents on demand for large datasets.

- **WebBaseLoader** – loads and extracts text content from web pages (URLs) using BeautifulSoup under the hood to parse HTML and extract visible text.
  - *When to use:* blogs, news articles, or public websites where content is primarily text-based and static.
  - *Limitations:* doesn't handle JavaScript-heavy pages well (use `SeleniumURLLoader` instead); loads only static HTML content, not what renders after page load.

- **CSVLoader** – loads CSV files into `Document` objects, one per row by default.

- **Other Document Loaders** – additional loaders exist for formats like Notion, JSON, databases, etc.

---

## 11. Text Splitters

### Text Splitting
Text Splitting is the process of breaking large chunks of text (articles, PDFs, HTML pages, books) into smaller, manageable pieces ("chunks") that an LLM can handle effectively.

**Why split text?**
- **Overcoming model limitations** – many embedding and language models have maximum input size constraints; splitting lets you process documents that would otherwise exceed these limits.
- **Downstream tasks** – text splitting improves nearly every LLM-powered task.
- **Optimizing computational resources** – smaller chunks are more memory-efficient and allow better parallelization.

### Splitting Strategies

**1. Length-Based Text Splitting**
Splits text purely by character/token count, regardless of sentence or paragraph boundaries.

**2. Text-Structured Based**
Splits along natural text structure — paragraphs, sentences, lines.

**3. Document-Structured Based**
Splits according to the structure of the document format itself (e.g., Markdown headers, code blocks, HTML tags).

**4. Semantic Meaning Based**
Groups text by topical/semantic similarity rather than fixed size, e.g., separating a passage about farming, a passage about cricket, and a passage about terrorism into distinct chunks based on meaning shifts.

---

## 12. Vector Stores

### Why Vector Stores?
Vector stores allow semantic search over unstructured data (e.g., finding a movie by plot description rather than exact keyword match) by comparing embedding vectors instead of raw text.

### What are Vector Stores?
A vector store is a system designed to store and retrieve data represented as numerical vectors.

**Key Features:**
1. **Storage** – retains vectors and associated metadata, in-memory or on-disk.
2. **Similarity Search** – retrieves vectors most similar to a query vector.
3. **Indexing** – provides data structures/methods for fast similarity search on high-dimensional vectors (e.g., approximate nearest neighbor lookups).
4. **CRUD Operations** – manage the lifecycle of vector data (create, read, update, delete).

**Use Cases:**
1. Semantic search
2. RAG
3. Recommender systems
4. Image / multimedia search

### Vector Store vs. Vector Database

| | Vector Store | Vector Database |
|---|---|---|
| Scope | Lightweight library/service focused on storing embeddings + similarity search | Full-fledged database system for storing and querying vectors |
| Features | May lack transactions, rich query languages, RBAC | Distributed architecture, durability (replication/backup), metadata schemas & filters, ACID/near-ACID guarantees, auth & security |
| Best for | Prototyping, smaller-scale applications | Production environments with significant scale, large datasets |
| Examples | FAISS | Milvus, Qdrant, Weaviate |

> A vector database is effectively a vector store with extra database features (clustering, scaling, security, metadata filtering, durability).

### Vector Stores in LangChain
- **Supported stores:** FAISS, Pinecone, Chroma, Qdrant, Weaviate, etc.
- **Common interface:** A uniform Vector Store API lets you swap one backend (e.g., FAISS) for another (e.g., Pinecone) with minimal code changes.
- **Metadata handling:** Most vector stores allow attaching metadata (timestamps, authors, etc.) to each document, enabling filter-based retrieval.

### Chroma Vector Store
Chroma is a lightweight, open-source vector database that is especially friendly for local development and small- to medium-scale production needs.

---

## 13. Retrievers

### What are Retrievers?
A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.

- There are multiple types of retrievers.
- **All retrievers in LangChain are Runnables.**

### Types of Retrievers

- **Wikipedia Retriever** – queries the Wikipedia API to fetch relevant content for a given query.

- **Vector Store Retriever** – the most common type of retriever; searches and fetches documents from a vector store based on semantic similarity using vector embeddings.

- **Maximal Marginal Relevance (MMR)** – an information retrieval algorithm designed to reduce redundancy in retrieved results while maintaining high relevance to the query.

- **Multi-Query Retriever** – generates multiple reformulations of the query to improve retrieval recall.

- **Contextual Compression Retriever** – an advanced retriever that improves retrieval quality by compressing documents after retrieval, keeping only the content relevant to the user's query.

---

## 14. RAG (Retrieval-Augmented Generation)

### What is RAG?
RAG is a technique that combines information retrieval with language generation, where a model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.

**Benefits of RAG:**
1. Use of up-to-date information
2. Better privacy
3. No limit on document size

**Related concepts:**
- **In-Context Learning** – a core capability of LLMs (GPT-3/4, Claude, Llama) where the model learns to solve a task purely by seeing examples in the prompt, without updating its weights.
- **Emergent Property** – a behavior or ability that suddenly appears in a system once it reaches a certain scale or complexity, even though it wasn't explicitly programmed or expected from the individual components.

### Understanding RAG — The 4 Stages

**1. Indexing** — preparing your knowledge base so it can be efficiently searched at query time. Sub-steps:
   1. **Document Ingestion** – load your source knowledge into memory.
   2. **Text Chunking** – break large documents into small, semantically meaningful chunks.
   3. **Embedding Generation** – convert each chunk into a dense vector (embedding) that captures its meaning.
   4. **Storage in a Vector Store** – store the vectors along with the original chunk text + metadata in a vector database.

**2. Retrieval** — the real-time process of finding the most relevant pieces of information from the pre-built index, based on the user's question.

**3. Augmentation** — combining the retrieved documents (relevant context chunks) with the user's query to form a new, enriched prompt for the LLM.

**4. Generation** — the final step, where the LLM uses the user's query plus the retrieved/augmented context to generate a response.

---

## 15. RAG using LangChain

### Roadmap of Improvements to a Basic RAG Pipeline

1. **UI-based enhancements**
2. **Evaluation**
   - Ragas
   - LangSmith
3. **Indexing**
   - Document ingestion
   - Text splitting
   - Vector store
4. **Retrieval**
   - *Pre-Retrieval:* query rewriting using an LLM, multi-query generation, domain-aware routing
   - *During Retrieval:* MMR, hybrid retrieval, reranking
   - *Post-Retrieval:* contextual compression
5. **Augmentation**
   - Prompt templating
   - Answer grounding
   - Context window optimization
6. **Generation**
   - Answer with citation
   - Guardrailing
7. **System Design**
   - Multimodal
   - Agentic
   - Memory-based

---

## 16. Tools

### What is a Tool?
A tool is just a Python function (or API) that is packaged in a way the LLM can understand and call when needed.

**How tools fit into the agent ecosystem:**
An AI agent is an LLM-powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal.

### Built-in Tools
A built-in tool is a tool that LangChain already provides — pre-built, production-ready, and requiring minimal or no setup. You don't have to write the function logic yourself; you just import and use it.

### Custom Tools
A custom tool is a tool that you define yourself.

**Ways to create custom tools:**
- **Structured Tool** – a special type of tool where the input follows a structured schema, typically defined using a Pydantic model.
- **BaseTool** – the abstract base class for all tools in LangChain. It defines the core structure and interface any tool must follow, whether a simple one-liner or a fully customized function. All other tool types (e.g., `@tool`, `StructuredTool`) are built on top of `BaseTool`.

### Toolkits
A toolkit is a collection (bundle) of related tools that serve a common purpose, packaged together for convenience and reusability.

*Example:* `GoogleDriveToolKit` — a toolkit bundling several Google Drive–related tools.

---

## 17. Tool Calling

- **Tool Binding** – attaching a set of tools to a model so it knows what's available to call.
- **Tool Calling** – the model decides, based on the input, whether and which tool to call, along with the arguments.
- **Tool Execution** – actually running the selected tool with the arguments produced by the model and returning the result.

**Example:** Currency Conversion Tool — a worked example of binding, calling, and executing a custom currency-conversion tool from an LLM.

---

## 18. Agents

### What are AI Agents?
Systems where an LLM autonomously reasons, decides, and takes actions (via tools) to accomplish a goal — rather than just producing a single response.

### 1. ReAct
ReAct is a design pattern used in AI agents that stands for **Reasoning + Acting**. It allows an LLM to interleave internal reasoning ("Thought") with external actions (like tool use) in a structured, multi-step process. Instead of generating an answer in one go, the model thinks step by step, deciding what it needs to do next, and optionally calls tools (APIs, calculators, web search, etc.) to help it.

### 2. Agent & Agent Executor
- **Agent** – the reasoning component that decides which action/tool to take next.
- **Agent Executor** – the runtime loop that executes the agent's chosen actions, feeds results back to the agent, and repeats until a final answer is produced.

### 3. Creating an Agent
Steps for defining an agent: choose an LLM, bind tools, define a prompt/reasoning strategy (e.g., ReAct).

### 4. Creating an Agent Executor
Wrapping the agent in an executor loop that handles tool invocation and iteration until completion.

### 5. Flow Chart
High-level control flow: **User Query → Agent (Thought) → Tool Call → Observation → Agent (Thought) → ... → Final Answer.**

### 6. Example
A worked, end-to-end example of building and running a LangChain agent.

---

## Notes on This Document

These notes were compiled from a personal LangChain course (OneNote export, ~143 pages) covering fundamentals through agentic RAG systems. Several original pages consisted primarily of diagrams, flowcharts, or code screenshots that don't carry over to plain text — those sections are represented here by their headings/topics only. For the full visual detail (diagrams, code screenshots, flowcharts), refer to the original PDF.