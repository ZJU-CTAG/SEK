import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    reasoning_effort: str = "medium",
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    if model.startswith("o1-") or model.startswith("o3-") or model.endswith("-reasoner"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        kwargs.pop("temperature")
        kwargs["reasoning_effort"] = reasoning_effort
    
    return client.chat.completions.create(
        model=model,
        messages=message,
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret