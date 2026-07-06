# Enterprise RAG-based Knowledge Assistant

A RAG-based generative AI project designed to help organizations retrieve, summarize, and utilize internal knowledge more efficiently.
This project integrates unstructured data such as documents, messages, and voice-based text data, and connects them to an LLM-based response and report generation workflow.

## Overview

This project was developed as part of a global capstone project.
The goal was to design a system that reduces the time spent searching for scattered internal information and supports business decision-making through retrieval-based generative AI.

Instead of building a simple chatbot, this project focuses on designing a full data-AI pipeline: collecting enterprise data, preprocessing it, storing it in a searchable format, retrieving relevant context, and generating responses or reports based on the retrieved information.

## Key Features

* Collection and preprocessing of unstructured enterprise data
* STT-based conversion of voice data into text
* Document chunking and embedding generation
* Vector database-based document retrieval
* LLM-based question answering and report generation
* Streamlit-based prototype interface
* Business use cases such as work summary, investment report generation, and decision-support materials

## Data Pipeline

1. Collect internal data such as documents, messages, and voice records
2. Convert voice data into text using STT
3. Preprocess and split text data into searchable chunks
4. Generate embeddings for each text chunk
5. Store embedded data in a vector database
6. Retrieve relevant documents based on user queries
7. Generate answers, summaries, or reports using an LLM

## Tech Stack

* Python
* Streamlit
* LLM
* Embedding
* Vector Database
* STT
* RAG Pipeline

## Project Structure

```text
Global_Capstone
├── Data
├── pipelines
├── vectorstores
├── main.py
├── streamlit_app.py
└── requirements.txt
```

## My Contribution

* Designed the overall RAG-based service scenario
* Structured the data flow from unstructured data collection to LLM response generation
* Organized voice, document, and message data into a searchable knowledge format
* Contributed to the presentation materials and business use case design
* Proposed how RAG can reduce repetitive information search and improve internal knowledge utilization

## Relevance

This project is related to data engineering and AI solution design because it focuses on transforming unstructured enterprise data into a searchable and usable knowledge base.
It demonstrates experience in designing a data pipeline that connects preprocessing, embedding, vector search, and generative AI response generation to support business decision-making.
