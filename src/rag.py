#from src.llm import call_groq

#def run_rag(query, retriever):
 #   docs = retriever.invoke(query)

  #  context = "\n\n".join([doc.page_content for doc in docs])

   # prompt = f"""
#Answer the question ONLY using the context below.
#If the answer is not in the context, say "I don't know".

#Context:
#{context}

#Question:
#{query}
#"""

 #   return call_groq(prompt)
from src.llm import call_groq

def run_rag(query, retriever, memory):
    history_context = memory.get_context()

    
    rewrite_prompt = f"""
Given the conversation history and the latest question,
rewrite the question so it is fully self-contained.

Conversation:
{history_context}

Question:
{query}

Rewritten question:
"""
    rewritten_query = call_groq(rewrite_prompt)

    docs = retriever.invoke(rewritten_query)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{rewritten_query}
"""
    answer = call_groq(final_prompt)

    memory.add(query, answer)

    return answer
