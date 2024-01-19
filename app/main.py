from fastapi import FastAPI
from app.service import LLMChain
from app.schema import ChatItem, AnswerResponse

llm_chain = LLMChain("atom92/medical_lama_ultra", 256, 0.5, 0.95, 1.5)


app = FastAPI()


@app.post("/chat")
async def get_chat_answer(chat_item: ChatItem):
    answer = await llm_chain.llm_answer(chat_item.question, chat_item.history)
    return AnswerResponse(answer=answer)
