from fastapi import FastAPI
from service import LLMChain
from schema import HistoryItem, ChatItem, AnswerResponse

llm_chain = LLMChain("abc", 256, 0.4, 0.95, 1.1)


app = FastAPI()


@app.post("/chat")
async def get_chat_answer(chat_item: ChatItem):
    answer = await llm_chain.llm_answer(chat_item.question, chat_item.history)
    return AnswerResponse(answer=answer)
