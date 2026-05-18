import ollama

# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'
LANGUAGE_MODEL = 'ministral-3:3b'

DATASET_PATH = './dataset/cat-facts.txt'

TOP_K = 3
SIMILARITY_THRESHOLD = 0.45


# ============================================================
# LOAD DATASET
# ============================================================

print('Loading dataset...')

with open(DATASET_PATH, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file.readlines() if line.strip()]

print(f'Loaded {len(dataset)} entries')


# ============================================================
# VECTOR DATABASE
# ============================================================

# Each item:
# {
#     "text": "...",
#     "embedding": [...]
# }

VECTOR_DB = []


def generate_embedding(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response['embeddings'][0]


def add_chunk_to_database(chunk):
    embedding = generate_embedding(chunk)

    VECTOR_DB.append({
        'text': chunk,
        'embedding': embedding
    })


print('\nBuilding vector database...')

for chunk in dataset:
    add_chunk_to_database(chunk)

print(f'Vector DB contains {len(VECTOR_DB)} entries')


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def retrieve(query, top_k=TOP_K):
    query_embedding = generate_embedding(query)

    similarities = []

    for item in VECTOR_DB:
        similarity = cosine_similarity(
            query_embedding,
            item['embedding']
        )

        if similarity >= SIMILARITY_THRESHOLD:
            similarities.append({
                'text': item['text'],
                'similarity': similarity
            })

    similarities.sort(
        key=lambda x: x['similarity'],
        reverse=True
    )

    return similarities[:top_k]


# ============================================================
# CHAT LOOP
# ============================================================

print('\nRAG chatbot is ready!')
print('Type "exit" to quit.\n')

while True:

    user_query = input('Enter your message: ')

    if user_query.lower() == 'exit':
        break

    # --------------------------------------------------------
    # RETRIEVE RELEVANT KNOWLEDGE
    # --------------------------------------------------------

    retrieved_knowledge = retrieve(user_query)

    print('\nRetrieved knowledge:')

    if not retrieved_knowledge:
        print('No relevant knowledge found.')

    for item in retrieved_knowledge:
        print(f"[{item['similarity']:.4f}] {item['text']}")

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = '\n'.join([
        f"- {item['text']}"
        for item in retrieved_knowledge
    ])

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the retrieved context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided knowledge base."

Retrieved Context:
{context}
"""

    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_query
            }
        ],
        stream=True
    )

    print('\nAssistant: ', end='')

    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)

    print('\n')