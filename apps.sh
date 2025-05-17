mkdir -p results/apps

# MODEL_TYPE=mixtral
# MODEL_SIZE=ins


MODEL_TYPE=llama-3.1
MODEL_SIZE=70b-ins

# apps-introductory
# apps-interview
# apps-competition

with_keywords=1


TASK=apps-introductory
CUDA_VISIBLE_DEVICES=3,4,5,6 python generate.py \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --greedy \
    --root outputs \
    --dataset $TASK \
    --backend vllm \
    --tp 4\
    --evalperf_type instruct \
    --with_refine true 

# # CUDA_VISIBLE_DEVICES=1,2,3,4 python zero_shot_cot_generate.py \
# #     --model_type $MODEL_TYPE \
# #     --model_size $MODEL_SIZE \
# #     --greedy \
# #     --root outputs \
# #     --dataset $TASK \
# #     --backend vllm \
# #     --tp 4\
# #     --evalperf_type instruct \
# #     --zero_shot_cot 1

# # # sleep 5

TASK=apps-interview

CUDA_VISIBLE_DEVICES=3,4,5,6 python generate.py \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --greedy \
    --root outputs \
    --dataset $TASK \
    --backend vllm \
    --tp 4 \
    --evalperf_type instruct \
    --with_refine true
    # --zero_shot_cot 1

# sleep 5

TASK=apps-competition
CUDA_VISIBLE_DEVICES=3,4,5,6 python generate.py \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --greedy \
    --root outputs \
    --dataset $TASK \
    --backend vllm \
    --tp 4 \
    --evalperf_type instruct \
    --with_refine true
    # --zero_shot_cot 1

# TASK=apps-competition
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     # --with_keywords 1
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1
# MODEL_TYPE=mixtral
# MODEL_SIZE=ins
# MODEL_TYPE=llama-3.1
# MODEL_SIZE=70b-ins
# TASK=apps-introductory

# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4\
#     --evalperf_type instruct 

# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# TASK=apps-interview


# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# TASK=apps-competition
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords $with_keywords\

# MODEL_TYPE=mixtral
# MODEL_SIZE=ins
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     # --with_keywords 0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords $with_keywords





TASK=apps-interview
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1





# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# for api 
# python generate.py \
#     --model_type  deepseek-coder  \
#     --greedy \
#     --root outputs \
#     --dataset apps-introductory \
#     --backend openai \
#     --base_url https://api.deepseek.com \
#     --evalperf_type instruct \
#     --with_keywords 1

# python generate.py \
#     --model_type  deepseek-coder  \
#     --greedy \
#     --root outputs \
#     --dataset apps-interview \
#     --backend openai \
#     --base_url https://api.deepseek.com \
#     --evalperf_type instruct \
#     --with_keywords 1

# python zero_shot_cot_generate.py \
#     --model_type  deepseekcoder-v2  \
#     --model_size ins\
#     --greedy \
#     --root outputs \
#     --dataset apps-introductory \
#     --backend openai \
#     --base_url http://localhost:9091/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot true 

# python zero_shot_cot_generate.py \
#     --model_type  deepseekcoder-v2  \
#     --model_size ins\
#     --greedy \
#     --root outputs \
#     --dataset apps-interview \
#     --backend openai \
#     --base_url http://localhost:9091/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot true 


# python zero_shot_cot_generate.py \
#     --model_type  deepseekcoder-v2  \
#     --model_size ins\
#     --greedy \
#     --root outputs \
#     --dataset apps-competition \
#     --backend openai \
#     --base_url http://localhost:9091/v1 \
#     --evalperf_type instruct \
    # --zero_shot_cot true 

# python zero_shot_cot_generate.py\
#     --model_type gpt\
#     --model_size 3.5-turbo\
#     --greedy \
#     --root outputs \
#     --dataset apps-introductory \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot 1



# python zero_shot_cot_generate.py \
#     --model_type gpt\
#     --model_size 3.5-turbo\
#     --greedy \
#     --root outputs \
#     --dataset apps-interview \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot 1

# python zero_shot_cot_generate.py \
#     --model_type gpt\
#     --model_size 3.5-turbo\
#     --greedy \
#     --root outputs \
#     --dataset apps-competition \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot 1

# python generate.py \
#     --model_type gpt\
#     --model_size 3.5-turbo\
#     --greedy \
#     --root outputs \
#     --dataset apps-competition \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot 1

# apps-introductory
# apps-interview
# apps-competition



# TASK=apps-interview
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# TASK=apps-introductory
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1

# TASK=apps-competition
# CUDA_VISIBLE_DEVICES=4,5,6,7 python generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
#     --with_keywords 1