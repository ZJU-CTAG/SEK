import os
from typing import List
from tqdm import tqdm
import openai

from bigcodebench.gen.util.openai_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt,make_prompt
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import concurrent_call

class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, reasoning_effort="medium", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
    
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 1, task_type = None
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        flag = 'Analyze the given code problem. Try to extract the keywords from the code problem. For each identified keyword:'
        if flag in prompts[0]:
            messages = make_prompt(prompts[0],'keyword',None,self.instruction_prefix)
        else:
            task_type = 'code'
            messages = [make_raw_chat_prompt(
                task_prompt=prompt,
                subset=self.subset,
                split=self.split,
                instruction_prefix=self.instruction_prefix,
                response_prefix=self.response_prefix,
                tokenizer=None,
            ) for prompt in prompts]
        # use concurrency based batching for o1 and deepseek models
        if self.name.startswith("o1-") or self.name.startswith("o3-") or self.name.startswith("deepseek"):
            return self._codegen_batch_via_concurrency(messages, num_samples)

        return self._codegen_api_batch(messages, num_samples,task_type)

    def _codegen_api_batch(self, messages: List[str], num_samples: int,task_type:str) -> List[str]:

        if 'deep' in self.name.lower():
            client = openai.OpenAI(base_url=base_url)
        else:
            client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""), base_url=self.base_url
            )
            
        all_outputs = []
        if task_type=='code':
            messages = messages[0]
        ret = make_auto_request(
            client,
            message=messages,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            n=num_samples,
        )
        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)
        all_outputs.append(outputs)
        return all_outputs

    def _codegen_batch_via_concurrency(self, messages: List[str], num_samples: int) -> List[str]:
        batches = concurrent_call(
            num_samples, self._codegen_api_batch, messages, num_samples=1
        )
        return [[element for sublist in item for element in sublist] for item in zip(*batches)]

    def is_direct_completion(self) -> bool:
        return False