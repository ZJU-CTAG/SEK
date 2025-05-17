mkdir -p results/mbpp

MODEL_TYPE=llama-3.1
MODEL_SIZE=70b-ins
TASK=mbpp

# MODEL_TYPE=mixtral
# MODEL_SIZE=ins
# with_keywords=1

# CUDA_VISIBLE_DEVICES=1,2,3,4 python zero_shot_cot_generate.py \
#     --model_type $MODEL_TYPE \
#     --model_size $MODEL_SIZE \
#     --greedy \
#     --root outputs \
#     --dataset $TASK \
#     --backend vllm \
#     --tp 4 \
#     --evalperf_type instruct \
    # --zero_shot_cot $witsh_keywords
    #  --with_refine true


CUDA_VISIBLE_DEVICES=5,6 python generate.py \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --greedy \
    --root new_outputs \
    --dataset $TASK \
    --backend vllm \
    --tp 2 \
    --evalperf_type instruct \
    --with_keywords true
    # --with_keywords true
# for api 
# MODEL_TYPE=deepseek-coder
# python generate.py \
#     --model_type  deepseek-coder  \
#     --greedy \
#     --root outputs \
#     --dataset mbpp \
#     --backend openai \
#     --base_url https://api.deepseek.com \
#     --evalperf_type instruct \
    # --with_keywords true
#      --with_refine true


# python generate.py \
#     --model_type gpt\
#     --model_size 3.5-turbo \
#     --greedy \
#     --root outputs \
#     --dataset mbpp \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --with_keywords true 

# MODEL_TYPE=gpt
# MODEL_SIZE=3.5-turbo
# MODEL_TYPE=gpt
# MODEL_SIZE=4o-mini
# python generate.py \
#     --model_type $MODEL_TYPE\
#     --model_size $MODEL_SIZE\
#     --greedy \
#     --root outputs \
#     --dataset mbpp \
#     --backend openai \
#     --base_url https://api.chatanywhere.tech/v1 \
#     --evalperf_type instruct \
#     --with_refine true
    # --zero_shot_cot true 
# MODEL_TYPE=deepseekcoder-v2
# MODEL_SIZE=ins
# python zero_shot_cot_generate.py \
#     --model_type $MODEL_TYPE\
#     --model_size $MODEL_SIZE\
#     --greedy \
#     --root outputs \
#     --dataset mbpp \
#     --backend openai \
#     --base_url http://localhost:9091/v1 \
#     --evalperf_type instruct \
#     --zero_shot_cot true 


# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0
# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0_cot
# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0zero_shot_cot
SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0_keywords-1-rank
# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0_refine
# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0_keywords_refine
# SAVE_PATH=outputs/$TASK/${MODEL_TYPE}${MODEL_SIZE:+_$MODEL_SIZE}_temp_0.0_beam4


SAVE_SANTH_PATH=$SAVE_PATH-sanitized

evalplus.evaluate \
  --dataset $TASK \
  --samples $SAVE_PATH \
  --i-just-wanna-run


evalplus.sanitize --samples $SAVE_PATH

evalplus.evaluate \
  --dataset $TASK \
  --samples $SAVE_SANTH_PATH \
  --i-just-wanna-run

# evalplus.evaluate \
#   --dataset mbpp \
#   --samples outputs/mbpp/deepseek-coder_temp_0.0_keywords \
# #   --i-just-wanna-run
# evalplus.sanitize  --samples outputs/mbpp/deepseek-coder_temp_0.0_keywords
# evalplus.sanitize outputs/mbpp/gpt_3.5-turbo_temp_0.0
  # evalplus.sanitize outputs/mbpp/deepseek-coder_temp_0.0_cot
# evalplus.evaluate \
#   --dataset mbpp \
#   --samples  outputs/mbpp/deepseek-coder_temp_0.0_cot-sanitized\
#   --i-just-wanna-run

# # outputs/mbpp/deepseek-coder_temp_0.0_keywords
# # evalplus.evaluate \
# #   --dataset mbpp \
# #   --samples outputs/mbpp/deepseek-coder_temp_0.0_refine-sanitized\
# #   --i-just-wanna-run


# evalplus.sanitize  outputs/mbpp/deepseek-coder_temp_0.0_cot

# evalplus.evaluate \
#   --dataset mbpp \
#   --samples  outputs/mbpp/old/llama-3.1_70b-ins_temp_0.0\
#   --i-just-wanna-run