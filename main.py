# main.py
from agent.llm_manager import get_llm
from langchain_core.prompts import PromptTemplate


def main():
    prompt = PromptTemplate.from_template("Explain {topic} in simple terms.")
    llm = get_llm(temperature=0.7)
    chain = prompt | llm

    response = chain.invoke(
        {"topic": "the difference between AI and Machine Learning"})

    print("\n🧠 Response:\n", response.content if hasattr(
        response, 'content') else response)


if __name__ == "__main__":
    main()
