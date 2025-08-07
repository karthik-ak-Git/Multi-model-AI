# agent/llm_manager.py

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub, Together
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI as OpenRouterLLM
from langsmith.run_helpers import traceable


load_dotenv()


@traceable(name="LLM Provider Selector")
def get_llm(model_name=None, temperature=0.7):
    provider = os.getenv("AI_PROVIDER", "OLLAMA").upper()

    if provider == "OLLAMA":
        model = model_name or os.getenv("OLLAMA_MODEL", "mistral")
        print(f"[🔗] Using Ollama model: {model}")
        return ChatOllama(model=model, temperature=temperature)

    elif provider == "OPENROUTER":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in .env")
        print(f"[🔐] Using OpenRouter LLM")
        return OpenRouterLLM(api_key=api_key, temperature=temperature, base_url="https://openrouter.ai/api/v1")

    elif provider == "GOOGLE":
        print("[🔐] Using Google Gemini via langchain_google_genai")
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model=os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash"),
            temperature=temperature,
            max_tokens=None,
            max_retries=2,
            timeout=None
        )

    elif provider == "HUGGINGFACE":
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")
        print(f"[🔐] Using HuggingFace Hub LLM")
        return HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=token
        )

    elif provider == "TOGETHER":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Missing TOGETHER_API_KEY in .env")
        print(f"[🔐] Using Together.ai LLM")
        return Together(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            together_api_key=api_key,
            temperature=temperature
        )

    else:
        raise ValueError(f"Invalid AI_PROVIDER '{provider}' specified in .env")
