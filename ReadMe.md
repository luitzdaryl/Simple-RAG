## Code a Simple RAG from Scratch

- RAG: Retrieval-Augmented Generation

- A RAG system consists of 2 key components: 

    1- A retrieval model that fetches relevant information from an external knowledge source, which could be a Database, search engine, or any other information repository

    2- A language model that generates responses based on the retrieved knowledge

- There are several ways to implement RAG, including Graph RAG, Hybrid RAG, and Hierarchical RAG

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Let's build a Simple RAG from Scratch:

- The system will comprise the following components: 

  User --- [ask] ---> Embedding Model --- [search] ---> Vector Database --- [knowledge] ---> Chatbot --- [response] ---> User

- Embedding Model: A pre-trained language model that converts input into embeddings-vector representations that capture semantic meaning.     These vectors will be used to search for relevant information in the dataset.  

- Vector Database: A storage system for knowlegde and its corresponding embedding vectors. While there are many vector DB technologies like Qdrant, Pinecone, and pgvector, we will implement a simple in-memory database from scratch.

- Chatbot: A language model that generates responses based on retrieve knowledge. This can be any language model, such as LLama, Gemma, or GPT.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Indexing Phase

- This is the first step when creating RAG system. It generally involves breaking the dataset (or the document) into small chunks and calculating a vector representation for each chunk that can be efficiently searched during generation.

[Chunk-1] -------|
[Chunk-2] -------|
[Chunk-3] -------| --->  [Embedding-Model] --- > [Vector-Database]
   |             |
   |             |
[Chunk-N] -------|  

- The size of each chunk can vary depending on the dataset and the application.(Could be a paragraph or a sentence).
- After the indexing phase, each chunk with its corresponding embedding vector will be stored in the vector database. 
- Example: [Chunk] ---> Haiti is a beautiful country. [Embedding-Vector] ---> \[0.1, 0.04, -0.34, 0.21, ...\]

- To compare the similarity between two vectors, we can use Cosine Similarity, Eucclidean Distance, or other metrics. However, we will use Cosine Similarity in this example. 

For two vectors A and B, the Cosine Similarity Formula is:

- cos(θ)=A⋅B/∥A∥ ∥B∥​

Meaning of each part:

- A⋅B = dot product
- ∥A∥ = magnitude (Euclidean norm) of vector A
- ∥B∥ = magnitude of vector B

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Retrieval Phase

- Given an [Input-Query] from [User]. Then we will calculate the [Query-Vector] to represent the query and compare it against the vectors in the database to find the most relevant chunks.

- The result returned by the [Vector-Database] will contains top N most relevant chunks to the query. These chunks will be used by the [Chatbot] to generate response.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Let's code it anyway:

- We will use a simple Python implement of RAG
- We will use Ollama to run the models locally

	​

	​




