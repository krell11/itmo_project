from langchain_openai import ChatOpenAI


class LlmClient:
    def __init__(self, api_base: str, model_name: str, temperature: float, tools=None):
        self.llm = ChatOpenAI(
            base_url=api_base,
            model_name=model_name,
            openai_api_key="dummy_key",
            temperature=temperature,
            max_tokens=500
        )
        self.tools = tools
        self.agent_executor = None

    def with_structured_output(self, model_cls):
        """Creates a structured LLM output with the given model class"""
        return self.llm.with_structured_output(model_cls)
