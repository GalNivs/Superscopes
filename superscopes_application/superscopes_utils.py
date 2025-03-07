from flask import jsonify
from sentence_transformers import SentenceTransformer, util
import torch
torch.set_grad_enabled(False)

# Utilities
from general_utils import (
  ModelAndTokenizer,
  make_inputs,
  decode_tokens,
)
from patchscopes_utils import (
    set_hs_patch_hooks_neox,
    set_hs_patch_hooks_llama,
    set_hs_patch_hooks_gptj,
    remove_hooks
)

model_to_hook = {
    "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    "meta-llama/Llama-2-13b-chat-hf": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj
}

CURRENT_LLM = "meta-llama/Llama-2-13b-chat-hf"
CURRENT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

mt = None
model_cos = None


def _init():
    """Initialization function that loads neccessary context."""
    global mt, model_cos

    model_name = CURRENT_LLM

    if "13b" in model_name or "12b" in model_name:
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    my_device = torch.device("cuda:0")

    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        device=my_device,
    )

    mt.set_hs_patch_hooks = model_to_hook[model_name]
    mt.model.eval()

    # loading semantic-similarity trained model to judge for best results
    model_name_cos = CURRENT_SIMILARITY_MODEL
    model_cos = SentenceTransformer(model_name_cos)


def patchscope_interpret(vec, target_layer=0):
    """Interpretation of vectors using Patchscopes technique."""
    target_prompt = "Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South Korean multinational major appliance and consumer electronics corporation, x"

    # last token within target prompt
    target_idx = -1

    patch_config = {
        target_layer: [(target_idx, vec)]
    }

    patch_hooks = mt.set_hs_patch_hooks(
        mt.model, patch_config, module="hs", patch_input=False, generation_mode=True,
    )

    inp = make_inputs(mt.tokenizer, [target_prompt], device=mt.device)

    seq_len = len(inp["input_ids"][0])
    max_token_to_produce = 10
    output_toks = mt.model.generate(
        inp["input_ids"],
        max_length=seq_len + max_token_to_produce,
        pad_token_id=mt.model.generation_config.eos_token_id,
    )
    
    remove_hooks(patch_hooks)

    generations_patched =  mt.tokenizer.decode(output_toks[0][len(inp["input_ids"][0]):])
    
    return generations_patched


def grade_output(original_prompt, patchscopes_output):
    """Grading Patchscopes outputs using a model explicitly trained for semantic similarity."""
    embeddings = model_cos.encode([original_prompt, patchscopes_output], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    return score


def superscopes_analyze_vector(prompt, vec, target_layer=0):
    """
    Applying the Superscopes technique to analyze an input vector 'vec'.
    The amplification values are chosen carefully to give a good range of interpretation possiblities.
    At first we try the vector as-is and then in deltas of 3 we try the 3-15 amplification range.
    """
    amplify_vals = [1, 3, 6, 9, 12, 15]

    results = []

    for amp in amplify_vals:
        res = patchscope_interpret(vec * amp, target_layer=target_layer)
        grade = grade_output(prompt, res)
        results.append((res, amp, grade))

    return {'best_result': max(results, key=lambda x: x[2]), 'all_results': results}


def generate_and_extract_intermediate_values(prompt):
    """
    Running the model and generating intermediate representations in each layer for:
    1. residuals pre-mlp
    2. mlp outputs
    3. hidden states
    """
    store_hooks = []
    residual_pre_mlp_cache_ = []
    mlp_cache_ = []

    input_ids = make_inputs(mt.tokenizer, [prompt], device=mt.device)

    def store_mlp_hook(module, input, output):
        residual_pre_mlp_cache_.append(input[0][0])
        mlp_cache_.append(output[0])

    for layer in mt.model.model.layers:
        store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))

    generated = mt.model(**input_ids, output_hidden_states=True)

    hs_cache_ = [
        generated["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)
    ]

    remove_hooks(store_hooks)

    return residual_pre_mlp_cache_, mlp_cache_, hs_cache_


def superscopes_analyze(prompt, start_layer, end_layer, patch_target):
    """Superscopes util main loop."""
    residual_pre_mlp_cache_, mlp_cache_, hs_cache_ = generate_and_extract_intermediate_values(prompt)

    input_ids = make_inputs(mt.tokenizer, [prompt], device=mt.device)
    decoded = decode_tokens(mt.tokenizer, input_ids['input_ids'])[0]

    only_best = []
    all_options = []

    for tok in range(len(input_ids["input_ids"][0])):
        only_best_layers_data = []
        all_options_layers_data = []
        for layer in range(start_layer, end_layer + 1):
            if patch_target == '0':
                target_layer = 0
            else:
                target_layer = layer

            all_data = {
                'residual_pre_mlp': superscopes_analyze_vector(prompt, residual_pre_mlp_cache_[layer][tok], target_layer=target_layer),
                'mlp_output': superscopes_analyze_vector(prompt, mlp_cache_[layer][tok], target_layer=target_layer),
                'hidden_state': superscopes_analyze_vector(prompt, hs_cache_[layer][tok], target_layer=target_layer),
            }

            only_best_layers_data.append({
                'layer_name': f"Layer {layer}",

                'residual_pre_mlp_interpretation': all_data['residual_pre_mlp']['best_result'][0],
                'residual_pre_mlp_amp': all_data['residual_pre_mlp']['best_result'][1],

                'mlp_output_interpretation': all_data['mlp_output']['best_result'][0],
                'mlp_output_amp': all_data['mlp_output']['best_result'][1],

                'hidden_state_interpretation': all_data['hidden_state']['best_result'][0],
                'hidden_state_amp': all_data['hidden_state']['best_result'][1]
            })

            for i in range(len(all_data['residual_pre_mlp']['all_results'])):
                all_options_layers_data.append({
                'layer_name': f"Layer {layer} Amp={all_data['residual_pre_mlp']['all_results'][i][1]}",

                'residual_pre_mlp_interpretation': all_data['residual_pre_mlp']['all_results'][i][0],
                'residual_pre_mlp_amp': all_data['residual_pre_mlp']['all_results'][i][1],

                'mlp_output_interpretation': all_data['mlp_output']['all_results'][i][0],
                'mlp_output_amp': all_data['mlp_output']['all_results'][i][1],

                'hidden_state_interpretation': all_data['hidden_state']['all_results'][i][0],
                'hidden_state_amp': all_data['hidden_state']['all_results'][i][1]
            })

        only_best.append({
            'token': decoded[tok],
            'layers': only_best_layers_data
        })

        all_options.append({
            'token': decoded[tok],
            'layers': all_options_layers_data
        })

    return jsonify({
        'onlyBest': only_best,
        'showAll': all_options,
        'patchTargetUsed': patch_target
    })


# initialize the module by loading the neccessary models
_init()
