import json
import keyword
import pickle
import os,sys
from os import PathLike
from typing import TypedDict
import pickle
from modele import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
import re
sys.path.append('../')
from datasets import load_dataset

from prompt import prompt_1,prompt_2
from filter_keyword import KeywordFilter,extract_keywords,format_output
from evaluate import load
from index import add_index

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_MAPPING={
    'llama-3.1':{
        '70b-ins':"Meta-Llama-3.1-70B-Instruct"
    },
    'gpt':{
        '3.5-turbo':'gpt-3.5-turbo-ca',
        '4o-mini':'gpt-4o-mini'
    },
    'mixtral':{
        'ins':'Mixtral-8x22B-Instruct-v0.1'
    },
}
EXAMPLE_KEYWORD = {
    'mbpp':'assert',
    'humaneval':['example','>>>'],
    'apps':'-----Input-----'
}

class Text2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str
    docstring: str

def get_extract_prompt(task,dataset_name):
    prompt = task["prompt"]
    if 'apps' != dataset_name:
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
        task['keyword_prompt'] = prompt_1.format(
            query=docstring)
    else:
        task['keyword_prompt'] = prompt_1.format(
            query=prompt)
    return task




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

def process_keywords(text):
    flags = ['[Keyword]','[Keyword_3]:','[Keyword_2]:','[Keyword_1]:','[Formalized explanation]']
    for flag in flags:
        text = text.replace(flag,'')
    if text.strip().startswith(': '):
        text = text.strip().strip(':')
    return text

def process_output(output,task):
    output_lines = output.split('\n')
    task_lines = task['prompt'].split('\n')
    if 'mbpp' in task['task_id'].lower():
        for task_line in task_lines:
            if 'assert' in task_line:
                function_name_part = task_line.split('=')[0]
                function_name_part = function_name_part.split('assert')[-1]
                if function_name_part.startswith(' math.'):
                    function_name= function_name_part.split('(')[1]
                elif function_name_part.startswith(' set(') :
                    function_name = function_name_part.split('(')[1]
                elif ' not ' in function_name_part:
                    function_name = function_name_part.replace(' not ','')
                    function_name = function_name.split('(')[0]
                else:
                    function_name = function_name_part.split('(')[0]
    else:
        function_name = task['entry_point']
    for i,output_line in enumerate(output_lines):
        if 'def' in output_line:
            if function_name in output_line:
                return output
    for i,output_line in enumerate(output_lines):
        if 'def' in output_line:
            new_line = re.sub(r'def\s+(\w+)\s*\(',r'def '+function_name+r'(',output_line)
            output_lines[i] = new_line
            return '\n'.join(output_lines)

    return output
        


def concat_prompt(task,keyword,dataset_name,model_name):
    keywords = capture_keywords(keyword)
    constr='Analyze the following key terms and their relationships within the problem context:'
    task_type = dataset_name
    if task_type=='humaneval':
        constr = '    '+constr
        keywords = ['    '+k for k in keywords]
    res = [constr]
    if 'app' in task_type:
        if '---' in task['prompt']: 
            prefix = '-----'
            suffix = '-----'
        elif '##' in task['prompt']: 
            prefix = '##'
            suffix = ''
        else:
            prefix = ''
            suffix = ':'
        res = [prefix+'Keywords and Explanations'+suffix]
        res += ['Analyze the following key terms and their relationships within the problem context:']

        keywords = [k.strip() for k in keywords]

    if len(keywords)==0: keyword=[]
    res += keywords
    res = '\n'.join(res)
    prompt = task['prompt']
    example_keyword = EXAMPLE_KEYWORD[task_type]
    example_line_index = None
    try:
        if task_type=='mbpp':
            example_line_index='none'
            if example_keyword in prompt:
                example_index = prompt.index(example_keyword)
            else:
                prompt_lines = prompt.split('\n')
                for i, line in enumerate(prompt_lines):
                    if line.strip().lower().startswith('example') or line.strip().lower().startswith('for example'):
                        example_line_index = i
                        break
                    elif line.strip().lower().startswith('original test cases') or line.strip().lower().startswith('test case'):
                        example_line_index = i
                        break
                    elif line.strip().startswith('>>>'):
                        example_line_index = i
                        break
                    elif line.strip().startswith('[input/output] samples:'):
                        example_line_index = i
                        break

                example_index = prompt.index(prompt_lines[example_line_index])
        elif task_type=='humaneval':
            prompt_lines = prompt.split('\n')
            for i, line in enumerate(prompt_lines):
                if line.strip().lower().startswith('example') or line.strip().lower().startswith('for example'):
                    example_line_index = i
                    break
                elif line.strip().lower().startswith('original test cases') or line.strip().lower().startswith('test case'):
                        example_line_index = i
                        break
                elif line.strip().startswith('>>>'):
                    example_line_index = i
                    break
                elif line.strip().startswith('[input/output] samples:'):
                    example_line_index = i
                    break
            if example_line_index == None:
                example_keyword=task['entry_point']+'('
                for i,line in enumerate(prompt_lines):
                    if example_keyword in line and 'def' not in line:
                        example_line_index = i
                        break
                        

            if example_line_index != None:
                example_index = prompt.index(prompt_lines[example_line_index])
            else:
                example_index = prompt.rindex('"""')
        else:
            prompt_lines = prompt.split('\n')
            for i, line in enumerate(prompt_lines):
                if line.strip().strip('#').strip('-').lower().startswith('input'):
                    example_line_index = i
                    break
                elif line.strip().strip('#').strip('-').lower().startswith('example'):
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
        if example_line_index== None and task_type=='humaneval':
            example_index = prompt.rindex("'''")
    if len(keywords)==0:
        task['keyword_prompt'] = prompt
    else:
        if task_type=='humaneval':
            if example_line_index != None:
                prompt = prompt[: docstring_end]+ '\n' + res + '\n\n' + prompt[example_index:]
            else:
                prompt = prompt[: docstring_end]+ '\n' + res + '\n' + prompt[example_index:]
        elif task_type=='mbpp':
            prompt = prompt[: docstring_end]+ '\n' + res + '\n\n' + prompt[example_index:]
        else:
            if example_line_index != None:
                prompt = prompt.strip()+ '\n\n' + res 
            else:
                prompt = prompt.strip()+ '\n' + res 
        task['keyword_prompt'] = prompt
    return task,keywords


def codegen(
    target_path: PathLike,
    model: DecoderBase,
    dataset: str,
    greedy=False,
    n_samples=1,
    id_range=None,
    version="default",
    resume=True,
    with_keywords=False,
    keyword_filter=None,
):
    global add_index
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1
    

    with Progress(
        TextColumn(f"{dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus(version=version)
            dataset_name = 'humaneval'
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            dataset = get_mbpp_plus(version=version)
            dataset_name = 'mbpp'
        elif dataset in ['apps-interview','apps-competition','apps-introductory']:
            level = dataset.split('-')[-1]
            dataset_path = './apps/'+level
            os.makedirs(dataset_path,exist_ok=True)
            with open(dataset_path+f'/{level}.json') as f:
                dataset = json.load(f)
            dataset_index = []
            for key in dataset:
                dataset_index.append(dataset[key]['problem_id'])
            dataset_name = 'apps'

        dataset_res = [] 
            
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)
            
            

            data_type = 'call-based'
            if dataset_name == 'apps':
                try:
                    input_outpout = json.loads(task["input_output"])
                    fn_name = (
                        None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
                    )
                except ValueError:
                    fn_name = None
                if not fn_name:
                    data_type = 'stdio-based'
                    task['entry_point']=''
                else:
                    data_type = 'call-based'
                    task['entry_point']=fn_name
                    
                    
            keyword_output = ''
            if with_keywords:
                task = get_extract_prompt(task,dataset_name)
                keyword_output = model.codegen(
                    task["keyword_prompt"],
                    do_sample=False,
                    num_samples=1,
                    task_type='keyword',
                    dataset=dataset_name
                )[0]
                keyword_list = extract_keywords(keyword_output)
                filter_keyword_list = keyword_filter.rank_by_importance(keyword_list, task['prompt'], task['entry_point'])
                keyword_output = format_output(keyword_output, filter_keyword_list)

                task,keyword_output_list = concat_prompt(task, keyword_output,dataset_name,model.name)
                keyword_output_str = '\n'.join(keyword_output)
            

            if with_keywords:
                prompt_flag = 'keyword_prompt' 
            else:
                prompt_flag = 'prompt'

            sidx = n_samples - n_more_samples
            
            while sidx < n_samples:
                outputs = model.codegen(
                    task[prompt_flag],
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                    task_type='code',
                    dataset=dataset_name,
                    data_type=data_type
                )
                assert outputs, "No outputs from model!"
                impl_list = []
                for impl in outputs:
                    if dataset_name!='apps':
                        impl = process_output(impl,task)
                        with open(
                            os.path.join(target_path, p_name, f"raw.txt"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(task['prompt']+'\n'+impl)
                    else:
                        with open(
                            os.path.join(target_path, p_name, f"raw.txt"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(task[prompt_flag]+'\n'+impl)
                        flag = '```python'
                        if flag in impl:
                            impl = impl[impl.index(flag)+len(flag):]
                        flag = '```'
                        if impl.count(flag)>=2:
                            impl = impl[impl.index(flag)+len(flag):]
                            impl = impl[:impl.index(flag)]
                        elif impl.count(flag)==1:
                            impl = impl[:impl.rindex(flag)]
                        else:
                            impl = impl.strip()
                        impl_list.append(impl)
                    solution = (
                        task["prompt"] + impl if model.is_direct_completion() else impl
                    )
                    if target_path.endswith(".jsonl"):
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n" 
                            )
                    else:
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                            
                    sidx += 1
                dataset_res.append(impl_list)    
        if dataset_name=='apps':
            if len(dataset_res)!=0:
                with open(
                        os.path.join(target_path, f"result_list.txt"),
                        "w",
                        encoding="utf-8",
                        ) as f:
                            f.write(repr(dataset_res))
            else:

                dir_list = os.listdir(target_path)
                dir_list = [item for item in dir_list if 'APPS' in item]
                dataset_index = [int(item.split('_')[-1]) for item in dir_list]
                
                for i in range(len(dataset_index)):
                    temp_list = []
                    for j in range(n_samples):
                        with open(os.path.join(target_path, f"{dir_list[i]}",f"{j}.py"),'r') as f:
                            temp_list.append(f.read())
                    dataset_res.append(temp_list)

            metric = load("./apps_metric")
            results = metric.compute(predictions=dataset_res,indices=dataset_index, level=level, k_list=[n_samples], count_errors=True)
            with open(os.path.join(target_path, f"result.json"), "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

def main(
    model_type: str,
    model_size: str|None = None,
    dataset: str|None = None,
    root: str|None = None,
    bs: int = 1,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range = None,
    version: str = "default",
    backend: str = "vllm",
    base_url: str = None,
    tp: int = 2,
    evalperf_type: str = None,  # This is for EvalPerf
    jsonl_fmt: bool = False,
    with_keywords: bool = False,
):
    # assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"
    assert backend in ["vllm", "hf", "openai"]
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]
    global prompt_1
    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, dataset), exist_ok=True)
    instruction_prefix="Please provide a self-contained Python script that solves the following problem in a markdown code block:"

    response_prefix='Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:'


    if model_size!=None:
        model_path = MODEL_MAPPING[model_type][model_size]
    else:
        model_path=model_type
    
    keyword_dataset = load_dataset('json', data_files='./python_dataset.json',split='train')
    keyword_filter = KeywordFilter()
    corpus = keyword_dataset['instruction']  # Default
    keyword_filter.initialize_vectorizer(corpus)

    model_runner = make_model(
        model=model_path,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )

    identifier = model_path.replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"
    model_size = '' if model_size is None else f"_{model_size}"
    target_path = os.path.join(
        root,
        dataset,
        model_type
        + f"{model_size}"
        + f"_temp_{temperature}"
    )
    
    if with_keywords:
        target_path += "_keywords"
        
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)


    codegen(
        target_path=target_path,
        dataset=dataset,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        version=version,
        with_keywords=with_keywords,
        keyword_filter=keyword_filter,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)