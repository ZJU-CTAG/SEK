import os
import json
import argparse
from typing import Optional, Tuple

from bigcodebench.provider import DecoderBase, make_model
from bigcodebench.data import get_bigcodebench, write_jsonl
from bigcodebench.sanitize import sanitize
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from prompt import prompt_1
from filter_keyword import KeywordFilter,extract_keywords,format_output
from datasets import load_dataset


def get_extract_prompt(prompt):
    try:
        docstring_begin = prompt.index('"""')
        docstring_end = prompt.rindex('"""')
        docstring = prompt[docstring_begin: docstring_end+3]
    except ValueError:
        docstring_begin = prompt.index("'''")
        docstring_end = prompt.rindex("'''")
        docstring = prompt[docstring_begin: docstring_end+3]
    except:
        docstring = prompt

    docstring = docstring.strip()
    return prompt_1.format(query=docstring)
def process_keywords(text):
    flags = ['[Keyword]','[Keyword_3]:','[Keyword_2]:','[Keyword_1]:','[Formalized explanation]']
    for flag in flags:
        text = text.replace(flag,'')
    if text.strip().startswith(': '):
        text = text.strip().strip(':')
    return text
def capture_keywords(text):
    lines = text.strip().split('\n')
    keywords = []
    for line in lines:
        if len(process_keywords(line).strip())!=0:
            if 'Here are the top 3 keywords with their explanations' in line:
                continue
            elif 'Here is the analysis of the' in line:
                continue
            elif line.strip().startswith('Here '):
                continue
            keywords.append(f'    {process_keywords(line).strip()}')
    return keywords

def concat_prompt(task,keyword,split,model_name):
    keywords = capture_keywords(keyword)
    constr='Analyze the following key terms and their relationships within the problem context:'
    constr = '    '+constr
    keywords = ['    '+k for k in keywords]
    res = [constr]

    if len(keywords)==0: keyword=[]
    res += keywords
    res = '\n'.join(res)
    prompt = task[f'{split}_prompt']
    example_line_index = None
    try:
        prompt_lines = prompt.split('\n')
        for i, line in enumerate(prompt_lines):
            if line.strip().lower().startswith('example'):
                example_line_index = i
        if example_line_index != None:
            example_index = prompt.index(prompt_lines[example_line_index])
        else:
            example_index = -1
        docstring_end = prompt.rindex('"""')
    except:
        if "'''" in prompt:
            docstring_end = prompt.rindex("'''")
        else:
            docstring_end = -1
            
    if len(keywords)==0:
        task['keyword_prompt'] = prompt
    else:

        if example_line_index != None:
            prompt = prompt[: docstring_end]+ '\n' + res + '\n\n' + prompt[docstring_end:]
        else:
            prompt = prompt[: docstring_end]+ '\n' + res + '\n' + prompt[docstring_end:]

        task['keyword_prompt'] = prompt
    return task,keywords



def codegen(
    model: DecoderBase,
    target_path: str,
    split: str,
    subset: str,
    greedy: bool = False,
    strip_newlines: bool = False,
    n_samples: int = 1,
    id_range: Tuple[int, int] = None,
    resume: bool = True,
    batch_size: int = -1,
    with_keywords: bool = False,
    keyword_filter: KeywordFilter = None,
):
    with Progress(
        TextColumn(f"BigCodeBench--{split.capitalize()} ({subset.capitalize()}) •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
            
        dataset = get_bigcodebench(subset=subset)

        if model.is_direct_completion() and split == "instruct":
            raise Exception("Base model does not support direct completion for instruct tasks")
        
        # create target_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(target_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)
        
        batch_keyword_prompts = []
        batch_prompts = []
        batch_task_ids = []
        batch_nsamples = []
        batch_entry_points = []
        
        # Read existing data once if resuming
        task2nexist = {}
        if resume and os.path.exists(target_path):
            with open(target_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    task2nexist[item["task_id"]] = task2nexist.get(item["task_id"], 0) + 1
        
        for id_num, (task_id, task) in enumerate(p.track(dataset.items())):
            if id_range is not None:
                low, high = id_range
                if id_num < low:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue
                if id_num >= id_range[1]:
                    break
                    
            p_name = task_id.replace("/", "_")

            n_existing = task2nexist.get(task_id, 0)
            nsamples = n_samples - n_existing
            
            try:
                prompt = task[f"{split}_prompt"]
            except:
                raise Exception(f"Invalid split {split} for bigcodebench-{subset}")
            if strip_newlines:
                prompt = prompt.strip("\n")
            
            if nsamples>0 and with_keywords:
                keyword_prompt = get_extract_prompt(prompt)
                keyword_output = model.codegen(
                    [keyword_prompt],
                    do_sample=not greedy,
                    num_samples=1,
                )[0][0]
                keyword_list = extract_keywords(keyword_output)
                filter_keyword_list = keyword_filter.rank_by_importance(keyword_list, prompt, task['entry_point'])
                keyword_output = format_output(keyword_output, filter_keyword_list)

                task,keyword_output_list = concat_prompt(task, keyword_output,split,model.name)
                prompt = task['keyword_prompt']
            
            if nsamples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_nsamples.append(nsamples)
                batch_entry_points.append(task["entry_point"])
                
                log = f"Codegen: {p_name} @ {model}"
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"
                p.console.print(log)
            
            if (batch_size and len(batch_prompts) == batch_size) or id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1):
                if not batch_prompts and (id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1)):
                    break
                    
                
                outputs = model.codegen(
                    batch_prompts,
                    do_sample=not greedy,
                    num_samples=1,
                )
                                
                assert outputs, "No outputs from model!"
                
                samples = []
                for task_id, content, entry_point, nsamples, task_outputs in zip(batch_task_ids, batch_prompts, batch_entry_points, batch_nsamples, outputs):
                    if model.is_direct_completion():
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(content+completion, entry_point), raw_solution=content+completion)
                            for completion in task_outputs[:nsamples]
                        ])
                    else:
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(completion, entry_point), raw_solution=completion)
                            for completion in task_outputs[:nsamples]
                        ])

                print(f"Generated {len(samples)} samples")
                write_jsonl(target_path, samples, append=True)

                batch_prompts = []
                batch_task_ids = []
                batch_nsamples = []


def run_codegen(
    model: str,
    split: str,
    subset: str,
    root: str = "bcb_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    greedy: bool = False,
    reasoning_effort: str = "medium",
    strip_newlines: bool = False,
    direct_completion: bool = False,
    resume: bool = True,
    id_range: str = None,
    backend: str = "vllm",
    base_url: str = None,
    tp: int = 1,
    instruction_prefix: str = "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
    response_prefix: str ="Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
    skip_prefill: bool = False,
    revision: str = "main",
    trust_remote_code: bool = False,
    tokenizer_name: str = None,
    tokenizer_legacy: bool = False,
    with_keywords: bool = False,

):
    if 'gpt' in model and with_keywords:
        instruction_prefix="Please complete the python function below. The final complete version of your function must be returned within a code block. Here is the unfinished function:"

    if greedy or (temperature == 0 and n_samples == 1):
        temperature = 0
        n_samples = 1
        greedy = True
        print("Greedy decoding ON (--greedy): setting n_samples=1, temperature=0")

    if id_range is not None:
        id_range = [int(i) for i in id_range.split("-")]
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    os.makedirs(root, exist_ok=True)
    
    model_runner = make_model(
        model=model,
        backend=backend,
        subset=subset,
        split=split,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=not skip_prefill,
        base_url=base_url,
        tp=tp,
        revision=revision,
        trust_remote_code=trust_remote_code,
        direct_completion=direct_completion,
        tokenizer_name=tokenizer_name,
        tokenizer_legacy=tokenizer_legacy
    )
    
    extra = "-" + subset if subset != "full" else ""
    if reasoning_effort and model.startswith("o1-") or model.startswith("o3-") or model.endswith("-reasoner"):
        model = model + f"--{reasoning_effort}"

    if skip_prefill:
        identifier = model.replace("/", "--") + "--skip_prefill" + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    else:
        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    
    target_path = os.path.join(root, identifier)
    
    if not resume:
        os.remove(target_path)
        
    if with_keywords:
        target_path = target_path.replace('.jsonl','_keyword.jsonl')
        print('*'*10+'\n'+prompt_1+'\n'+'*'*10)


    
    keyword_dataset = load_dataset('json', data_files='../../python_dataset.json',split='train')
    keyword_filter = KeywordFilter()
    corpus = keyword_dataset['instruction']  
    keyword_filter.initialize_vectorizer(corpus)
    
    codegen(
        model=model_runner,
        target_path=target_path,
        split=split,
        subset=subset,
        greedy=greedy,
        strip_newlines=strip_newlines,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        batch_size=bs,
        with_keywords=with_keywords,
        keyword_filter=keyword_filter
    )

    return target_path


def main():
    from fire import Fire
    Fire(run_codegen)


if __name__ == "__main__":
    main()
