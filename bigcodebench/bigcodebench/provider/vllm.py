import os
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.inputs import TextPrompt
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import (
    extra_eos_for_direct_completion,
    make_prompt,
    make_raw_chat_prompt,
)

class VllmDecoder(DecoderBase):
    def __init__(self, name: str, dataset: str, tp: int, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", tp)),
            "dtype": self.dtype,
            "trust_remote_code": True,
            "revision": self.revision,
            'enforce_eager': True,
            'gpu_memory_utilization': 0.99,
        }
        if self.tokenizer_name is None:
            self.tokenizer_name = self.name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **kwargs, legacy=self.tokenizer_legacy)
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        # self.llm = LLM(model=name, max_model_len=self.max_new_tokens, **kwargs)
        self.llm = LLM(model=name, max_model_len=8192, **kwargs)
        self.llm.set_tokenizer(tokenizer=self.tokenizer)

    def is_direct_completion(self) -> bool:
        return self.tokenizer.chat_template is None or self.direct_completion

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200,task_type=None
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        flag = 'Analyze the given code problem. Try to extract the keywords from the code problem. For each identified keyword:'
        if flag in prompts[0]:
            prompts = make_prompt(prompts[0],'keyword',self.tokenizer,self.instruction_prefix)

        else:
            prompts = [
                make_raw_chat_prompt(
                    task_prompt=prompt,
                    subset=self.subset,
                    split=self.split,
                    instruction_prefix=self.instruction_prefix,
                    response_prefix=self.response_prefix,
                    prefill=self.prefill,
                    tokenizer=self.tokenizer,
                    direct_completion=self.direct_completion,
                )
                for prompt in prompts
            ]


        vllm_outputs = self.llm.generate(
            prompts,
            SamplingParams(
                n=num_samples,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
                skip_special_tokens=self.skip_special_tokens,
            ),
            # use_tqdm=True,
        )


        gen_strs = [[x.text.replace("\t", "    ") for x in output.outputs] for output in vllm_outputs]
        return gen_strs