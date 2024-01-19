from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import torch
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from app.schema import HistoryItem, ChatItem
from typing import List

load_dotenv(find_dotenv())


class LLMChain:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float = 1.1):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={0: "17GiB", 1: "17GiB"},
            offload_folder="/tmp/offload"
        )
        model.tie_weights()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        template = """As a general medical knowledge assistant, I aim to provide accurate and general information based on existing knowledge. However, always consult with a healthcare professional for personalized advice. Based on general medical literature,

        Current conversation:
        {history}
        Human: {input}
        AI Assistant:"""

        self._prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self._llm = HuggingFacePipeline(pipeline=pipe)
        # self._llm = OpenAI(temperature=0)

    async def llm_answer(self, query: str, history: List[HistoryItem]):
        conversation = ConversationChain(
            prompt=self._prompt,
            llm=self._llm,
            verbose=True,
            memory=self.generate_memory(history)
        )
        # print(self.window_memory.load_memory_variables({}))
        return conversation.run(query)

    @staticmethod
    def generate_memory(history: List[HistoryItem]):
        memory = ConversationBufferWindowMemory(ai_prefix="AI Assistant", k=2, return_messages=True)
        for history_item in history:
            memory.save_context({"input": history_item.human_message}, {"output": history_item.ai_message})
        return memory

