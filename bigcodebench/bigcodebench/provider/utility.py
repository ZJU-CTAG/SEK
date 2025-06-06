from typing import List
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from prompt import bigcodebench_shot_2,prompt_1
EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "bigcodebench":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_prompt(task_prompt,task_type,tokenizer,instruction_prefix):
    if task_type == 'keyword':
        icl_num=2
        temps = bigcodebench_shot_2
        temp = []


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

            return task_prompt

        else:     
            message = f"""{task_prompt.strip()}"""
            temp.append({"role": "user", "content": message})
            return temp
        
    

def make_raw_chat_prompt(
    task_prompt,
    subset: str,
    split: str, 
    instruction_prefix: str,
    response_prefix: str,
    tokenizer: AutoTokenizer,
    prefill: bool = True,
    direct_completion: bool = False,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer:
        if tokenizer.chat_template is None or direct_completion:
            return task_prompt
        
    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"
    if split == "complete":
        task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
        
    else:
        task_prompt = f"""\
{instruction_prefix}
{task_prompt.strip()}
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""

    temps = []
    
    if tokenizer:
        if prefill:
            temps.append({"role": "user", "content": task_prompt})
            temps.append({"role": "assistant", "content": response})
            task_prompt = tokenizer.apply_chat_template(
                temps,
                tokenize=False,
            ).split(_MAGIC_SPLITTER_)[0]
        else:
            temps.append({"role": "user", "content": task_prompt})
            task_prompt = tokenizer.apply_chat_template(
                temps,
                tokenize=False,
            ).split(_MAGIC_SPLITTER_)[0]
    else:
        temps.append({"role": "user", "content": task_prompt})
        return temps

    return task_prompt


def concurrent_call(n, callback, /, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(callback, *args, **kwargs) for _ in range(n)]
        return [future.result() for future in futures]