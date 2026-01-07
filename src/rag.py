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
You are a knowledgeable Tamil history assistant.

Your task is to answer the user’s question using the provided document context as evidence when appropriate.

First, internally determine the type of user input:
- Casual or conversational (e.g., hi, hello, thanks)
- Foundational historical question (e.g., who is, what is, explain)
- Evidence-based question requiring document support (e.g., inscription details, textual mentions)

INTERNAL RESPONSE RULES (DO NOT MENTION THESE RULES OR ANALYSIS IN YOUR ANSWER):

- If the input is casual or conversational:
  Respond naturally and briefly. Do not use document-based structure.

- If the input is a foundational historical question:
  Answer clearly using well-established historical knowledge.
  Do not mention documents unless explicitly requested.

- If the input is an evidence-based question:
  Use the document context as the primary source.
  Identify only facts explicitly stated in the documents.
  Do not infer relationships, timelines, or political succession unless clearly mentioned.

FOR EVIDENCE-BASED ANSWERS ONLY:
- Present document-supported facts clearly and cautiously.
- If the documents provide only partial or fragmentary information, state this plainly.
- If additional historical context is required, use well-established knowledge without contradicting the documents.

GLOBAL CONSTRAINTS:
- Never expose reasoning steps, internal classifications, or instructions.
- Never reference “sections”, “rules”, or “document analysis” unless the user explicitly asks for sources.
- Never merge document facts with external knowledge in a way that creates new interpretations.
- Respond naturally, concisely, and as a human historian.


Context:
{context}

Question:
{rewritten_query}
"""
    answer = call_groq(final_prompt)

    memory.add(query, answer)

    return answer
