from typing import Dict, Any
import time

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(
        self, 
        question: str, 
        top_k: int = 5, 
        min_score: float = 0.2, 
        stream: bool = False, 
        summarize: bool = False
    ) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            return {
                'question': question,
                'answer': "No relevant context found.",
                'sources': [],
                'summary': None,
                'history': self.history
            }

    
        context = "\n\n".join([doc['document'] for doc in results])
        sources = [{
            'source': doc['metadata'].get('filename', 'unknown'),
            'page': doc['metadata'].get('page', 'unknown'),
            'score': doc['distance'],
            'preview': doc['document'][:120] + '...'
        } for doc in results]

        prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {question}

Answer:"""

        if stream:
            print("Streaming answer:")
            for i in range(0, len(prompt), 80):
                print(prompt[i:i+80], end='', flush=True)
                time.sleep(0.05)
            print()

        response = self.llm.invoke([prompt])
        answer = response.content


        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }