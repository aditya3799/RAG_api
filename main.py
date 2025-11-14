from fastapi import FastAPI
from pydantic import BaseModel
from question import retrieve_hybrid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

#----------LLm setup and configuration ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("Gemini_api_key"),
    temperature=0.2
)

app=FastAPI()

class QueryRequest(BaseModel):
    question: str

#---------RAG response generation----------
def generate(context,question):
    prompt= """You are an aviation expert. You would require to answer the questions related to the boeing B737 aircraft.
    
            for each question use the below context retrived from the boeing manual of B737.
            context:{context}
            
            ---------------------------
            question:{question}
            ---------------------------

            Based on the above context provide a prompt and precise answer to the question.
            structure the response for a better readability.
            Provide the final answer in **one or two polished sentences**.
            DO NOT include step-by-step reasoning instead make 1-2 suggesting points or descriptions.
            DO NOT list instructions, bullets, or numbered points.
            DO NOT repeat or quote large text chunks from the context.
             DO NOT include reference text itself. Only cite page numbers.
            In case on context not matching queries include a bit of your response which might be suggestive.
            for each response provide some of the references to support your answer.

            """
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=prompt)

    response=llm.invoke(prompt_template.format(context=context, question=question))

    return response


def answer_query(question):
    context=retrieve_hybrid(question, top_k=5)
    combined_context="\n".join([f"Reference {i+1}:\n{item['text']}" for i, item in enumerate(context)])

    response=generate(combined_context,question)

    output={"response":response.content,
            "pages":[item['page'] for item in context]
            }
    
    return output


#--------- API endpoint ----------
@app.post("/query")
def rag_api(request: QueryRequest):

    result=answer_query(request.question)
    return result


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
