import gradio as gr
from ragen.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg
from verl import DataProto
from transformers import AutoTokenizer
import hydra
import os
import time
import asyncio

# --- Global agent object
agent_proxy = None

# --- Initialization function
@hydra.main(version_base=None, config_path="../../config", config_name="stream")
def init_agent(config):
    global agent_proxy
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    agent_proxy = LLMAgentProxy(config, actor_wg, tokenizer)

# --- Streaming rollout generator
def rollout_stream():
    lm_inputs = DataProto(batch=None, non_tensor_batch=None, meta_info={
        'eos_token_id': 151645,
        'pad_token_id': 151643,
        'recompute_log_prob': False,
        'do_sample': True,
        'validate': True
    })
    env_outputs = agent_proxy.val_es_manager.reset()
    agent_proxy.val_ctx_manager.reset_memory_managers()
    assert len(env_outputs) == 1

    for turn_idx in range(agent_proxy.config.agent_proxy.max_turn):
        lm_inputs = agent_proxy.val_ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
        lm_inputs.meta_info = {'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': True, 'validate': True}
        lm_outputs = agent_proxy.generate_sequences(lm_inputs)

        response_texts = lm_outputs.non_tensor_batch['response_texts']
        yield f"\n--- Turn {turn_idx + 1} ---\n"
        yield response_texts[0]

        env_inputs = agent_proxy.val_ctx_manager.get_env_inputs(lm_outputs)
        env_outputs = agent_proxy.val_es_manager.step(env_inputs)
        if len(env_outputs) == 0:
            break
        yield env_outputs[0]['history'][-1]['state']

# --- Gradio Streaming Setup
async def streaming_demo():
    stream = rollout_stream()
    output = ""
    for chunk in stream:
        output += chunk
        yield output

def main():
    init_agent()

    with gr.Blocks() as demo:
        output_box = gr.Textbox(label="Agent Output", lines=20)
        run_button = gr.Button("Run Agent")

        run_button.click(fn=streaming_demo, inputs=[], outputs=output_box)

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    main()
