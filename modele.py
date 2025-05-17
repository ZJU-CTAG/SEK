import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn

import openai

try:
    import anthropic

    from evalplus.gen.util import anthropic_request
except ImportError:
    warn("Anthropic decoder will not work. Fix by `pip install anthropic`")

# mistral.ai
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    warn("MistralAI decoder will not work. Fix by `pip install mistralai`")

import torch
from stop_sequencer import StopSequencer
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM , SamplingParams


from openai.types.chat import ChatCompletion

# from evalplus.gen.util import openai_request
import sys
sys.path.append('../')

from prompt import humaneval_shot_1,prompt_1,mbpp_shot_1,apps_shot_1,apps_shot_2,apps_shot_3

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ",'\ndef main(','\nprint(']
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert",'\ndef main(','\nprint(']
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def extract_original_prompt(prompt: str) -> str:
    flag = 'Be aware of the following keywords in the problem description:'
    return prompt.split(flag)[0].strip()

def make_request(
    client: openai.Client,
    message,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 1,
    n: int = 1,
    task_type:str='code',
    **kwargs
) -> ChatCompletion:
    system_msg = "You are a helpful assistant good at coding."
    if (
        kwargs.get("response_format", None)
        and kwargs["response_format"]["type"] == "json_object"
    ):
        system_msg = "You are a helpful assistant designed to output JSON."
    
    required_length = 0

    if task_type != 'keyword':
        message = [{'role': "system", "content": system_msg}]+message
        return client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=required_length,
            temperature=temperature,
            n=n,
            **kwargs
        )
    else:
        return client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=required_length,
            temperature=temperature,
            n=n,
            **kwargs
        )


def make_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer|None,
    task_type: str = "code",
    dataset: str = 'humaneval',
    data_type: str|None = 'call-based'
):
    if tokenizer:
        if tokenizer.chat_template is None:
            return task_prompt.strip()
    temps = []
    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"
    if task_type == 'keyword':
        if dataset=='humaneval':
            temps = humaneval_shot_1
            icl_num=2
        elif dataset=='mbpp':
            temps = mbpp_shot_1
            icl_num=1
        elif dataset=='apps':
            temps = apps_shot_1
            icl_num=2
        
        temp = []


        if task_type=='keyword':
            for line in temps[:icl_num]:
                user = {"role": "user", "content": prompt_1.format(query=line['problem'])}
                keywords = '\n'.join(line['keywords'])
                assistant = {"role": "assistant", "content":keywords}
                temp.append(user)
                temp.append(assistant)


        if tokenizer!=None:  
            task_prompt = f"""{task_prompt.strip()}"""
            response = f"""{_MAGIC_SPLITTER_}"""
            temp.append({"role": "user", "content": task_prompt})
            task_prompt = tokenizer.apply_chat_template(
            temp,
            tokenize=False,
            add_generation_prompt=True
        )   

        else:      
            message = f"""{task_prompt.strip()}"""
            temp.append({"role": "user", "content": message})
            return temp
    
    else:   
        task_prompt_t = """\
{instruction_prefix}
```
{query}
```
"""
        response_t = """\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""

        temp = []
        if dataset == 'apps':
            if data_type == 'call-based':
                temps = apps_shot_2
                task_prompt = task_prompt.strip()+'\n### Use Call-Based Format'
            elif data_type == 'stdio-based':
                temps = apps_shot_3
                task_prompt = task_prompt.strip()+'\n### Use Standard Input Format (read inputs with `input()`, write results with `print()`)'
            for line in temps:
                user = {"role": "user", "content": task_prompt_t.format(instruction_prefix=instruction_prefix,query=line['problem'])}
                assistant = {"role": "assistant", "content":response_t.format(response_prefix=response_prefix, _MAGIC_SPLITTER_=line['response'])}
                temp.append(user)
                temp.append(assistant)
        elif dataset == 'humaneval':
            temps = []
            for line in temps:
                user = {'role':'user','content':task_prompt_t.format(instruction_prefix=instruction_prefix,query=line['problem'])}
                assistant = {"role": "assistant", "content":
response_t.format(response_prefix=response_prefix, _MAGIC_SPLITTER_=line['response'])}
                temp.append(user)
                temp.append(assistant)
        elif dataset == 'mbpp':
            temps = []
            for line in temps:
                user = {"role": "user", "content": task_prompt_t.format(instruction_prefix=instruction_prefix,query=line['problem'])}
                assistant = {"role": "assistant", "content":response_t.format(response_prefix=response_prefix, _MAGIC_SPLITTER_=line['response'])}
                temp.append(user)
                temp.append(assistant)
        if tokenizer!=None:
            task_prompt = task_prompt_t.format(instruction_prefix=instruction_prefix, query=task_prompt.strip())
            temp.append({'role': 'user', 'content': task_prompt})
            response = response_t.format(response_prefix=response_prefix, _MAGIC_SPLITTER_=_MAGIC_SPLITTER_)
            temp.append({'role': 'assistant', 'content': response})
            task_prompt = tokenizer.apply_chat_template(
                temp,
                tokenize=False,
                add_generation_prompt=True
            ).split(_MAGIC_SPLITTER_)[0]
        else:  
            message = instruction_prefix
            message += f"\n```python\n{task_prompt.strip()}\n```"
            temp.append({"role": "user", "content": message})
            return temp
    return task_prompt




class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VllmDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, tp: int, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", tp)),
            "dtype": self.dtype,
            "trust_remote_code": True,
            "max_model_len": 2048,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.chat_template is None:
            self.eos += extra_eos_for_direct_completion(dataset)
        self.llm = LLM(model=name,gpu_memory_utilization=0.99,enforce_eager=True,**kwargs)   #gpu_memory_utilization=0.99,enforce_eager=True,

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is None

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        if num_samples==1:
            vllm_outputs = self.llm.generate(
                [prompt] * batch_size,
                SamplingParams(
                    temperature=0,
                    max_tokens=self.max_new_tokens,
                    top_p=0.95 if do_sample else 1.0,
                    stop=self.eos,
                ),
                use_tqdm=False,
            )
        else:
            vllm_outputs = self.llm.generate(
                [prompt] * batch_size,
                SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    top_p=0.95 if do_sample else 1.0,
                    stop=self.eos,
                ),
                use_tqdm=False,
            )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


class GeneralVllmDecoder(VllmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.eos += ["\n```\n"]
        print(f"EOS strings: {self.eos}")

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200 , task_type : str = 'code', dataset='humaneval',data_type:str='call-based'
    ) -> List[str]:
        prompt = make_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer,task_type,dataset,data_type
        )

        return VllmDecoder.codegen(self, prompt, do_sample, num_samples)
    

class HfTorchDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {}
        kwargs["device_map"] = "auto"
        kwargs["trust_remote_code"] = self.trust_remote_code
        kwargs["torch_dtype"] = getattr(torch, self.dtype)
        self.skip_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.chat_template is None:
            self.eos += extra_eos_for_direct_completion(dataset)

        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model = self.model.to(self.device)

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        stop_sequencer = StopSequencer(
            self.model,
            model_type="causal",  # or seq2seq
            tokenizer=self.tokenizer,
        )

        model = stop_sequencer.register_stop_texts(
            stop_texts=self.eos,
            input_length=input_tokens.size(-1),
        )

        outputs = model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs


class GenenralHfTorchDecoder(HfTorchDecoder):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eos += ["\n```\n"]
        print(f"EOS strings: {self.eos}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = make_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
        )
        return HfTorchDecoder.codegen(self, prompt, do_sample, num_samples)


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        if 'deep' in name.lower():
            self.client = openai.OpenAI(base_url=base_url)
        else:
            self.client = openai.OpenAI(base_url=base_url)
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, task_type : str = 'code',dataset='humaneval',data_type='call-based'
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)

        message = make_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, None, task_type,dataset,data_type
        )
            
        fmt = "json_object" if self.name == "gpt-4-1106-preview" else "text"

        
        ret = make_request(
            self.client,
            message=message,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            n=batch_size,
            response_format={"type": fmt},
            task_type=task_type,
        )

        outputs = []
        for item in ret.choices:
            content = item.message.content
            if fmt == "json_object":
                try:
                    json_data = json.loads(content)
                    if json_data.get("code", None) is not None:
                        outputs.append(prompt + "\n" + json_data["code"])
                        continue

                    print(f"'code' field not found in: {json_data}")
                except Exception as e:
                    print(e)
            outputs.append(content)

        return outputs

    def is_direct_completion(self) -> bool:
        return False


class MistralChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        kwargs = {}
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        else:
            self.temperature = 0

        batch_size = min(self.batch_size, num_samples)

        outputs = []
        for _ in range(batch_size):
            ret = self.client.chat(
                model=self.name,
                messages=[
                    ChatMessage(
                        role="user",
                        content=self.instruction_prefix
                        + f"\n```python\n{prompt.strip()}\n```",
                    )
                ],
                max_tokens=self.max_new_tokens,
                **kwargs,
            )

            outputs.append(ret.choices[0].message.content)

        return outputs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def is_direct_completion(self) -> bool:
        return False


class AnthropicMessageDecoder(AnthropicDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            message = anthropic_request.make_auto_request(
                client=self.client,
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": self.instruction_prefix
                        + f"\n```python\n{prompt.strip()}\n```\n",
                    }
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop_sequences=["\n```\n", "\nif "],
            )
            outputs.append(message.content[0].text)

        return outputs


def make_model(
    model: str,
    backend: str,
    dataset: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    tp=1,
    base_url=None,
    instruction_prefix=None,
    response_prefix=None,
):
    if backend == "vllm":
        return GeneralVllmDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
            tp=tp,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "hf":
        return GenenralHfTorchDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            dataset=dataset,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            base_url=base_url,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "mistral":
        return MistralChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "anthropic":
        return AnthropicMessageDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )