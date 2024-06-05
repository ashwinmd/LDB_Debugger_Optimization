from .py_generate import PyGenerator
# from .model import CodeLlama, ModelBase, GPT4, GPT35, StarCoder
from .model import ModelBase, GPT4, GPT35, GPT4o, Groq_Llama_Big, Groq_Llama_Small

def model_factory(model_name: str, port: str = "", key: str = "") -> ModelBase:
    if "gpt-4o" in model_name:
        return GPT4o(key)
    elif model_name == "gpt-3.5-turbo-0613":
        return GPT35(key)
    elif model_name == "llama3-8b-8192":
        return Groq_Llama_Small(key)
    elif model_name == "llama3-70b-8192":
        return Groq_Llama_Big(key)
    #elif model_name == "starcoder":
     #   return StarCoder(port)
    #elif model_name == "codellama":
     #   return CodeLlama(port)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
