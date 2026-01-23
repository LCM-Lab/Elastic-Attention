from typing import Optional, Tuple, List
import torch
from transformers import AutoTokenizer, StaticCache
from transformers.models.llama.modeling_llama import Cache, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    logging,
)
import flashinfer
import time

try:
    from elasticattn.src.Xattention import Xattention_prefill
except:
    print("Xattention Import Fail")
try:
    from elasticattn.src.Minference import Minference_prefill
except:
    print("Minference Prefill Import Fail")
try:
    from elasticattn.src.Fullprefill import Full_prefill
except:
    print("Full Prefill Import Fail")
try:
    from elasticattn.src.Flexprefill import Flexprefill_prefill
except:
    print("Flex Prefill Import Fail")
from elasticattn.src.utils import *

import os
import numpy as np
import json
import functools

from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def forward_eval(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Performs a forward pass of the attention layer with optimized prefill mechanisms.

    This function integrates various optimizations, including fused attention,
    efficient caching, and rotary embeddings, to enhance inference speed.

    Parameters:
    - hidden_states (torch.Tensor): The input hidden states of shape (batch_size, seq_len, hidden_dim).
    - attention_mask (torch.Tensor, optional): The attention mask tensor, which defines
    which positions should be attended to.
    - position_ids (torch.LongTensor, optional): The position indices of tokens in the sequence.
    - past_key_value (Cache, optional): Cached key-value tensors for faster decoding.
    - output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
    - use_cache (bool, optional): Whether to use caching for faster inference. Defaults to False.
    - cache_position (torch.LongTensor, optional): The position index used in caching.
    - position_embeddings (Tuple[torch.Tensor, torch.Tensor], optional): Precomputed RoPE
    embeddings (cos and sin components).
    - **kwargs: Additional arguments for flexibility.

    Returns:
    - Tuple[torch.Tensor, Optional[torch.Tensor]]:
    The attention output tensor and attention weights (if enabled).
    """
    if self.fastprefillconfig.print_detail:
        start_time = time.time()
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads
        if hasattr(self, "num_heads")
        else self.config.num_attention_heads,
        self.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)
    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        reshape_time = time.time() - start_time
        print(f"     Reshape took: {reshape_time:.6f} seconds")

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.fastprefillconfig.print_detail:
        start_time = time.time()
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if isinstance(past_key_value, StaticCache):
        key_states = key_states[
            :, :, : min(cache_position[-1] + 1, key_states.shape[2]), :
        ]
        value_states = value_states[
            :, :, : min(cache_position[-1] + 1, value_states.shape[2]), :
        ]

    _, _, k_len, _ = key_states.shape
    _, _, q_len, _ = query_states.shape
    decoding = q_len != k_len and q_len == 1
    if not decoding:
        key_states = repeat_kv(
            key_states,
            self.num_key_value_groups
            if hasattr(self, "num_key_value_groups")
            else self.config.num_attention_heads // self.config.num_key_value_heads,
        ).to("cuda")
        value_states = repeat_kv(
            value_states,
            self.num_key_value_groups
            if hasattr(self, "num_key_value_groups")
            else self.config.num_attention_heads // self.config.num_key_value_heads,
        ).to("cuda")
    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        past_kv_time = time.time() - start_time
        print(f"     Past KV update and repeat took: {past_kv_time:.6f} seconds")

    if self.fastprefillconfig.print_detail:
        start_time = time.time()
        print(f"q length: {q_len} k length: {k_len}")
    stride = self.fastprefillconfig.stride
    if not decoding:
        if self.fastprefillconfig.metric == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
            ).transpose(1, 2)
        elif self.fastprefillconfig.metric == "xattn":
            if isinstance(self.fastprefillconfig.threshold, torch.Tensor):
                attn_output = Xattention_prefill(
                    query_states,
                    key_states,
                    value_states,
                    stride,
                    norm=1,
                    threshold=self.fastprefillconfig.threshold[self.layer_idx],
                    use_triton=True,
                )
            else:
                attn_output = Xattention_prefill(
                    query_states,
                    key_states,
                    value_states,
                    stride,
                    norm=1,
                    threshold=self.fastprefillconfig.threshold,
                    use_triton=True,
                )
        elif self.fastprefillconfig.metric == "full":
            attn_output = Full_prefill(
                query_states, key_states, value_states, attention_mask=attention_mask
            )
        elif self.fastprefillconfig.metric == "minfer":
            attn_output = Minference_prefill(
                query_states, key_states, value_states, adaptive_budget=0.3
            )
    else:
        if key_states.device != query_states.device:
            key_states = key_states.to(query_states.device)
        if value_states.device != query_states.device:
            value_states = value_states.to(query_states.device)

        value_states = value_states.squeeze(0).contiguous()
        query_states = query_states.squeeze(0).squeeze(1)
        key_states = key_states.squeeze(0).contiguous()
        attn_output = flashinfer.single_decode_with_kv_cache(
            query_states, key_states, value_states, kv_layout="HND"
        )
        attn_output = attn_output.unsqueeze(0).unsqueeze(2)

    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        attn_time = time.time() - start_time
        print(f"     Attention computation took: {attn_time:.6f} seconds")

    if self.fastprefillconfig.print_detail:
        start_time = time.time()
    if attn_output.size() != (
        bsz,
        self.num_heads
        if hasattr(self, "num_heads")
        else self.config.num_attention_heads,
        q_len,
        self.head_dim,
    ):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads if hasattr(self, 'num_heads') else self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    del query_states
    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        post_attn_time = time.time() - start_time
        print(f"     Post-attention processing took: {post_attn_time:.6f} seconds")

    return attn_output, None


class FastPrefillConfig(dict):
    """
    Configuration class for FastPrefill, which provides flexible settings for optimizing
    prefill computations in transformer models.

    Attributes:
    - threshold (float or torch.Tensor, optional): The threshold for selecting relevant attention blocks.
    - print_detail (bool): Whether to print detailed timing and debugging information.
    - stride (int): Determines the level of fused attention computation (e.g., 16, 8, or 4).
    - metric (str): Defines the type of prefill mechanism used ('xattn', 'full', 'minfer', 'flex').

    Methods:
    - __init__: Initializes the configuration with user-defined or default values.
    """

    def __init__(
        self,
        threshold: float = None,
        print_detail: bool = False,
        stride=16,
        metric="xattn",
    ):
        """
        Initialize the configuration with default or user-provided values.
        """
        super().__init__()
        self.print_detail = print_detail
        self.metric = metric
        self.stride = stride
        if threshold is not None:
            self.threshold = torch.ones((32, 32)).to("cuda") * threshold
        else:
            if stride == 16:
                self.threshold = torch.tensor(llama_fuse_16)
            elif stride == 8:
                self.threshold = torch.tensor(llama_fuse_8)
            elif stride == 4:
                self.threshold = torch.tensor(llama_fuse_4)
        self.threshold = self.threshold.to("cuda")


def load_attn_pattern_new(attn_load_dir, sink_size=None, recent_size=None):
    if attn_load_dir.endswith(".tsv"):
        path = attn_load_dir
    else:
        path = os.path.join(attn_load_dir, "full_attention_heads.tsv")
    full_attention_heads = np.loadtxt(
        path,
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    if sink_size is None:
        config = json.load(open(os.path.join(attn_load_dir, "config.json")))
        sink_size = config["sink_size"]
        recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size


def format_chat(
    message, include_system=False, system_message="You are a helpful assistant."
):
    if include_system:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


def call_api(func, limit=5, pause=10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if (
                "rate limit" in str(e).lower()
                or "rate_limit" in str(e).lower()
                or "quota" in str(e).lower()
                or "429" in str(e)
            ):
                logger.info(
                    f"Rate limit exceeded, waiting {pause} secs and retrying..."
                )
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                output = None
                break
    return output


class LLM:
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    def prepare_inputs(self, test_item, data):
        raise NotImplementedError("prepare_inputs not implemented for LLM")

    def generate(self, inputs=None, prompt=None, **kwargs):
        raise NotImplementedError("generate not implemented for LLM")


class OpenAIModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        import openai
        import tiktoken

        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/") + 1 :]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def prepare_inputs(self, test_item, data):
        buffer = 100
        # we don't include system message to stay consistent with other models
        prompt = format_chat(
            data["user_template"].format(**test_item),
            include_system=False,
        )
        inputs = "\n".join(
            [f"Role: {x['role']}\nContent: {x['content']}" for x in prompt]
        )
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        max_length = self.max_length
        if max_length > 128000:
            logger.warning(
                f"max_length {max_length} is greater than 128000, setting to 128000"
            )
            max_length = 128000

        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (
                max_length - self.generation_max_length - buffer
            )
            new_context = self.tokenizer.decode(
                self.tokenizer.encode(test_item["context"])[:-truncate_length]
            )
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), include_system=False
            )
        return prompt

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """

    def generate(
        self,
        inputs=None,
        prompt=None,
        system_message="You are a helpful assistant",
        **kwargs,
    ):
        if inputs is None:
            inputs = format_chat(
                prompt, include_system=True, system_message=system_message
            )

        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None


class AnthropicModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        from anthropic import Anthropic, AnthropicVertex

        if "vertex" in model_name:
            # region defaults to env var CLOUD_ML_REGION and project_id defaults to ANTHROPIC_VERTEX_PROJECT_ID
            self.model = AnthropicVertex()
            model_name = model_name[model_name.index("/") + 1 :]
        else:
            # remember to set ANTHROPIC_API_KEY environment variable (the default)
            self.model = Anthropic()

        self.tokenizer = self.model.get_tokenizer()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.do_sample = do_sample
        self.stops = None
        if stop_newline:  # claude does not support newline
            pass

    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item),
            include_system=False,
        )
        inputs = "\n".join(
            [f"Role: {x['role']}\nContent: {x['content']}" for x in prompt]
        )
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (
                self.max_length - self.generation_max_length - buffer
            )
            tokens = self.tokenizer.encode(test_item["context"])
            new_context = test_item["context"][
                : tokens.offsets[-truncate_length - 1][1]
            ]
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item),
                include_system=False,
            )
        return prompt

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """

    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, include_system=False)

        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        # Note: in the original paper, we used this system message:
        # system="You are a helpful assistant. Make sure your output does not contain new lines."
        # To be consistent with the other models, and for future compability, we remove the system message
        # We don't expect this to make a significant difference in the results
        func = functools.partial(
            self.model.messages.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop_sequences=self.stops,
            **kwargs,
        )
        output = call_api(func, pause=20)

        if output is not None:
            return {
                "output": output.content[0].text,
                "input_len": output.usage.input_tokens,
                "output_len": output.usage.output_tokens,
                "input_text": inputs,
            }
        return None


class GeminiModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        import google.generativeai as genai

        # default env var GOOGLE_API_KEY
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        import vertexai

        vertexai.init()  # make sure to set the env var appropriately
        from vertexai.preview.tokenization import get_tokenizer_for_model

        self.model = genai.GenerativeModel(model_name)
        self.tokenizer = get_tokenizer_for_model(model_name)
        self.model_name = model_name

    def prepare_inputs(self, test_item, data):
        prompt = data["prompt_template"].format(**test_item)
        buffer = 100
        inputs = self.tokenizer.compute_tokens(prompt).token_info_list[0].tokens
        input_len = len(inputs)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (
                max_length - self.generation_max_length - buffer
            )
            # not the most pretty way of doing this but it works...
            # the documentation doesn't provide an official way to truncate
            new_context = self.tokenizer._sentencepiece_adapter._tokenizer.decode(
                self.tokenizer.compute_tokens(test_item["context"])
                .token_info_list[0]
                .token_ids[:-truncate_length]
            )
            test_item["context"] = new_context
            prompt = data["prompt_template"].format(**test_item)

        return prompt

    def generate(self, inputs=None, prompt=None, **kwargs):
        import google.generativeai as genai

        if inputs is None:
            inputs = prompt

        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.generation_max_length,
        )
        func = functools.partial(
            self.model.generate_content,
            contents=inputs,
            generation_config=generation_config,
        )
        output = call_api(func, pause=15)
        if output is not None:
            try:
                # can probably check the output for errors but it's not well documented
                output.text
            except Exception as e:
                logger.error(f"Error in output: {output}; {e}")
                return None

            return {
                "output": output.text,
                "input_len": output.usage_metadata.prompt_token_count,
                "output_len": output.usage_metadata.candidates_token_count,
                "input_text": inputs,
            }
        return None


class TogetherModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        from transformers import AutoTokenizer
        from together import Together

        # default env var TOGETHER_API_KEY
        self.model = Together()
        # should change this to be more flexible in the future lol
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-405B-Instruct"
        )
        self.model_name = model_name.replace("togetherapi/", "")

    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item),
            system_message=data.get("system_message", "You are a helpful assistant."),
        )
        tokens = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        input_len = len(tokens)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (
                max_length - self.generation_max_length - buffer
            )
            context_tokens = self.tokenizer(
                test_item["context"], return_offsets_mapping=True
            )
            new_context = test_item["context"][
                : context_tokens["offset_mapping"][-truncate_length][0]
            ]

            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item),
                system_message=data.get(
                    "system_message", "You are a helpful assistant."
                ),
            )
        return prompt

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """

    def generate(
        self,
        inputs=None,
        prompt=None,
        system_message="You are a helpful assistant",
        **kwargs,
    ):
        if inputs is None:
            inputs = format_chat(
                prompt, include_system=True, system_message=system_message
            )

        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None


def tokenize(
    sample, data, tokenizer, max_length, generation_max_length, use_chat_template=False
):
    def format_input(sample):
        if use_chat_template:
            chat = format_chat(
                data["user_template"].format(**sample),
                include_system=False,
                system_message=data.get(
                    "system_message", "You are a helpful assistant."
                ),
            )
            try:
                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                chat = format_chat(
                    data["user_template"].format(**sample),
                    include_system=False,
                )
                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )

            tokenized_input = tokenizer(
                [prompt], return_tensors="pt", add_special_tokens=False
            )
        else:
            prompt = data["prompt_template"].format(**sample)
            tokenized_input = tokenizer([prompt], return_tensors="pt")
        return tokenized_input

    if "Phi3SmallTokenizer" in str(type(tokenizer)):
        buffer = (
            64 if max_length == 131072 else 0
        )  # there is some problem with their rotary emb implementation
    else:
        buffer = 0

    tokenized_input = format_input(sample)
    if tokenized_input.input_ids.size(1) > max_length - generation_max_length - buffer:
        truncate_length = tokenized_input.input_ids.size(1) - (
            max_length - generation_max_length - buffer
        )

        # handle non-fast hf tokenizers (e.g., phi-3-small)
        if isinstance(tokenizer, PreTrainedTokenizer) and not tokenizer.is_fast:
            context_tokens = tokenizer(sample["context"])
            new_context = tokenizer.decode(
                context_tokens["input_ids"][:-truncate_length]
            )
        else:
            context_tokens = tokenizer([sample["context"]], return_offsets_mapping=True)
            new_context = sample["context"][
                : context_tokens["offset_mapping"][0][-truncate_length][0]
            ]

        sample["context"] = new_context
        tokenized_input = format_input(sample)
    return tokenized_input


class HFModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
            AutoConfig,
        )

        model_kwargs = {}
        from pkg_resources import parse_version

        if parse_version(transformers.__version__) <= parse_version("4.34.1"):
            model_kwargs["use_flash_attention_2"] = True
        else:
            model_kwargs["attn_implementation"] = kwargs.get(
                "attn_implementation", "flash_attention_2"
            )
        if "recurrentgemma" in model_name or "yarn" in model_name.lower():
            model_kwargs = {}

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "elasticattn" in kwargs and kwargs["elasticattn"]:
            logger.warning(
                "elasticattn is enabled. Loading custom sparse attention model..."
            )
            is_llama = any(
                "llama" in arch.lower() for arch in getattr(config, "architectures", [])
            )
            is_qwen = any(
                "qwen" in arch.lower() for arch in getattr(config, "architectures", [])
            )
            is_phi = any(
                "phi" in arch.lower() for arch in getattr(config, "architectures", [])
            )
            if is_llama:
                from elasticattn.training.modeling_flash_llama import (
                    PawLlamaForCausalLM,
                    PawLlamaConfig,
                )

                CustomModelClass = PawLlamaForCausalLM
                CustomConfigClass = PawLlamaConfig
                logger.info("Detected LLaMA architecture. Using PawLlamaForCausalLM.")
            elif is_qwen:
                from elasticattn.training.modeling_flash_qwen import (
                    PawQwen3ForCausalLM,
                    PawQwen3Config,
                )

                CustomModelClass = PawQwen3ForCausalLM
                CustomConfigClass = PawQwen3Config
                logger.info("Detected Qwen architecture. Using PawQwen3ForCausalLM.")
            elif is_phi:
                from elasticattn.training.modeling_flash_phi import (
                    PawPhi3ForCausalLM,
                    PawPhi3Config,
                )

                CustomModelClass = PawPhi3ForCausalLM
                CustomConfigClass = PawPhi3Config
                logger.info("Detected Phi architecture. Using PawPhi3ForCausalLM.")
            else:
                raise ValueError(
                    f"Unsupported architecture for elasticattn: {config.architectures}. "
                    "Only LLaMA and Qwen are supported."
                )
            AutoModelForCausalLM.register(CustomConfigClass, CustomModelClass)
            if "rope_theta" in kwargs and kwargs["rope_theta"] is not None:
                logger.info(f"Override rope theta to {kwargs['rope_theta']}")
                config.rope_theta = kwargs["rope_theta"]
            self.model = CustomModelClass.from_pretrained(
                model_name,
                config=config,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                device_map="auto",
                trust_remote_code=True,
            )
            if (
                hasattr(config, "suggested_threshold")
                and config.suggested_threshold is not None
            ):
                threshold = config.suggested_threshold
                logger.info(
                    f"Using suggested_threshold={threshold} for deterministic sparse attention"
                )
            else:
                threshold = 0.5
                logger.warning(
                    "No suggested_threshold found. Using default threshold=0.5."
                )
            self.model.set_threshold_for_deterministic(threshold)
            self.model.eval()

            self.chunk_prefilling = kwargs["duoattn_chunk_prefilling"]
            if self.chunk_prefilling is not None:
                logger.warning(
                    f"Using chunk prefilling (size={self.chunk_prefilling}) for DuoAttention!"
                )
        else:
            if "rope_theta" in kwargs and kwargs["rope_theta"] is not None:
                logger.info(f"Override rope theta to {kwargs['rope_theta']}")
                config.rope_theta = kwargs["rope_theta"]

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                device_map="auto",
                trust_remote_code=True,
                **model_kwargs,
            )

        self.special_settings = {}
        if "duoattn" in kwargs and kwargs["duoattn"] is not None:
            logger.warning("Using DuoAttention patch for evaluation!")
            logger.warning(
                "Note that when using DuoAttention, we use eager attention implementation for compatibility"
            )
            from elasticattn.src.utils import sparsify_attention_heads
            from elasticattn.src.utils import enable_duo_attention_eval

            duoattn_path = kwargs["duoattn"]
            duoattn_sparsity = kwargs["duoattn_sparsity"]
            attn_heads, sink_size, recent_size = load_attn_pattern_new(
                duoattn_path,
                sink_size=kwargs["duoattn_sink"],
                recent_size=kwargs["duoattn_sliding"],
            )
            if kwargs["duoattn_flipping"]:
                logger.warning(
                    "| Flipping the duoattn pattern (for debugging purposes)"
                )
                attn_heads = 1 - attn_heads

            attn_heads, sparsity = sparsify_attention_heads(
                attn_heads, sparsity=duoattn_sparsity
            )

            logger.warning(f"True sparsity: {sparsity}")
            logger.warning(f"Sparse attention heads: {attn_heads.shape}")

            enable_duo_attention_eval(
                self.model,
                attn_heads,
                sink_size=kwargs["duoattn_sink"],
                recent_size=kwargs["duoattn_sliding"],
            )
            self.chunk_prefilling = kwargs["duoattn_chunk_prefilling"]
            if self.chunk_prefilling is not None:
                logger.warning(
                    f"Using chunk prefilling (size={self.chunk_prefilling}) for DuoAttention!"
                )

            self.special_settings["method"] = "duo"
            self.special_settings["method_params"] = {
                "sparsity": duoattn_sparsity,
                "sink_size": kwargs["duoattn_sink"],
                "recent_size": kwargs["duoattn_sliding"],
            }
        elif (
            "fastprefill_metric" in kwargs and kwargs["fastprefill_metric"] is not None
        ):
            logger.warning(f"Using XAttention or FlexPrefill patch for evaluation!")
            fastprefillconfig = FastPrefillConfig(
                threshold=kwargs.get("fastprefill_threshold", 0.95),
                print_detail=kwargs.get("fastprefill_print_detail", False),
                stride=kwargs.get("fastprefill_stride", 16),
                metric=kwargs.get("fastprefill_metric", "xattn"),
            )
            method = kwargs["fastprefill_metric"]
            # fastprefillconfig info
            logger.warning(f"{method} is running!!!")
            if "llama" in model_name.lower():
                import types
                import functools
                from elasticattn.src.utils import llama_causal_model_forward

                # multiple gpus inference using Accelerate
                if isinstance(self.model.forward, functools.partial):
                    self.model.forward.__wrapped__ = types.MethodType(
                        llama_causal_model_forward, self.model
                    )
                else:
                    self.model.forward = types.MethodType(
                        llama_causal_model_forward, self.model
                    )
                from elasticattn.src.utils import llama_mlp_forward
                from transformers.models.llama.modeling_llama import (
                    LlamaMLP,
                )

                for idx, m in self.model.named_modules():
                    if isinstance(m, LlamaMLP):
                        m.forward = types.MethodType(llama_mlp_forward, m)
            for layer in self.model.model.layers:
                layer.self_attn.fastprefillconfig = fastprefillconfig
                layer.self_attn.forward = forward_eval.__get__(layer.self_attn)
            self.chunk_prefilling = (
                kwargs["duoattn_chunk_prefilling"]
                if kwargs["duoattn_chunk_prefilling"] is not None
                else None
            )
        elif "minference" in kwargs and kwargs["minference"] is not None:
            from minference import MInference
            from minference.modules.kvcompression import method_to_cache_obj

            logger.warning(f"**** USING {kwargs['minference']} for evaluation! ****")
            if kwargs["minference"] == "minference":
                logger.warning("Using MInference for evaluation!")
                minference = MInference(
                    attn_type="minference", model_name=kwargs["minference_model_name"]
                )
                self.model, _ = minference(self.model)
                raise NotImplementedError("MInference is not supported!")
            elif kwargs["minference"] in ["pyramidkv", "snapkv"]:
                logger.warning("Using PyramidKV for evaluation!")
                minference = MInference(
                    attn_type="dense",
                    kv_type=kwargs["minference"],
                    model_name=kwargs["minference_model_name"],
                    attn_kwargs={
                        "window_size": kwargs["minference_window_size"],
                        "max_capacity_prompt": kwargs["minference_max_capacity_prompt"],
                        "compress_group_kvs": kwargs["minference_compress_group_kvs"],
                    },
                )
                self.model, config = minference(self.model)

                # Ready a cache
                config.num_layers = self.model.config.num_hidden_layers
                self.special_settings["past_key_values"] = (
                    method_to_cache_obj[kwargs["minference"]],
                    config,
                )
                self.special_settings["is_pyramid_snapkv"] = True

                # If we have a sparsity value, use that
                if kwargs["minference_sparsity"] is not None:
                    self.special_settings["sparsity"] = kwargs["minference_sparsity"]
                else:
                    self.special_settings["total_prefill_budget"] = kwargs[
                        "minference_max_capacity_prompt"
                    ]  # - kwargs["minference_window_size"]? Maybe not
                self.special_settings["local_window_size"] = kwargs[
                    "minference_window_size"
                ]
                self.special_settings["method"] = "pyramid_snap_kv"
                self.special_settings["method_params"] = {
                    "window_size": kwargs["minference_window_size"],
                    "max_capacity_prompt": kwargs["minference_max_capacity_prompt"],
                    "do_patch": kwargs.get("minference_chunking_patch", False),
                    "compress_group_kvs": kwargs["minference_compress_group_kvs"],
                }
            elif kwargs["minference"] == "l2":
                logger.warning("Using L2 for evaluation!")
                minference = MInference(
                    attn_type="dense",
                    kv_type=kwargs["minference"],
                    model_name=kwargs["minference_model_name"],
                    attn_kwargs={
                        "num_skip_layers": 2,
                        "max_capacity_total": kwargs["minference_max_capacity_prompt"],
                        "num_local_tokens": kwargs["minference_window_size"],
                    },
                )
                self.model, config = minference(self.model)

                # Ready a cache
                config.num_layers = self.model.config.num_hidden_layers
                self.special_settings["past_key_values"] = (
                    method_to_cache_obj[kwargs["minference"]],
                    config,
                )
                self.special_settings["method"] = "l2"
                self.special_settings["method_params"] = {
                    "num_skip_layers": 2,
                    "num_total_layers": self.model.config.num_hidden_layers,
                    "max_capacity_total": kwargs["minference_max_capacity_prompt"],
                }
            else:
                raise ValueError(f"Invalid minference type: {kwargs['minference']}")

            self.chunk_prefilling = kwargs.get("minference_chunk_prefilling", None)
        else:
            logger.warning("Using vanilla HF model for evaluation!")
            self.chunk_prefilling = None

        if kwargs.get("torch_compile", True):
            logger.warning("Using torch compile for evaluation!")
            self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = (
            [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        )
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set(
                    [
                        self.tokenizer.convert_tokens_to_ids(stop_token)
                        for stop_token in stop
                    ]
                    + stop_token_ids
                )
            )
            if "llama" in model_name.lower():
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device
        self.disable_prefill = False

        if "gemma" in model_name.lower():
            self.disable_prefill = True
            logger.warning(
                "gemma models cannot prefill with past kvs due to cache implementation, need to change the code manually if you need to prefill"
            )

    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item,
            data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            use_chat_template=self.use_chat_template,
        )

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=self.max_length - self.generation_max_length,
                truncation=True,
                padding=True,
            )

        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and not self.disable_prefill:
            # prefill without calculating the logits (save memory for large vocab models)
            extra = {}
            if "jamba" in str(type(self.model)).lower():
                from transformers.models.jamba.modeling_jamba import (
                    HybridMambaAttentionDynamicCache,
                )

                cache = HybridMambaAttentionDynamicCache(
                    self.model.config,
                    inputs.input_ids.shape[0],
                    self.model.dtype,
                    device=self.model.device,
                )
                extra = {"past_key_values": cache}

            # We should do chunked prefilling if (1) there is a valid chunk prefilling size, or (2) if we have a special setting that tells us that the last `k` tokens should be prefilled separately
            method = self.special_settings.get("method", "vanilla")

            if self.chunk_prefilling is not None:
                past_key_values = None
                prefilling_input_ids = inputs.input_ids[..., :-1]

                prefill_indices = list(
                    range(0, prefilling_input_ids.size(1), self.chunk_prefilling)
                )
                prefill_sizes = [self.chunk_prefilling] * (len(prefill_indices) - 1)
                prefill_sizes.append(
                    prefilling_input_ids.size(1)
                    - self.chunk_prefilling * (len(prefill_indices) - 1)
                )

                # Prefill
                for i, (index, size) in enumerate(zip(prefill_indices, prefill_sizes)):
                    # For pyramid/snapkv: (1) use the readied cache at the beginning, and (2) apply the special settings
                    if (
                        past_key_values is None
                        and "past_key_values" in self.special_settings
                    ):
                        past_key_values_class, past_key_values_config = (
                            self.special_settings["past_key_values"]
                        )
                        past_key_values = past_key_values_class(past_key_values_config)
                    if self.special_settings.get("is_pyramid_snapkv", False):
                        # Now, apply the special settings
                        # The chunks are not equally sized, so recalculate budget for this chunk
                        prefill_budget = int(
                            (1 - self.special_settings["sparsity"]) * size
                        )
                        if self.special_settings["method_params"][
                            "do_patch"
                        ] and index + size < prefilling_input_ids.size(1):
                            # We will patch an extra window_size tokens, so we need to add that to the prefill budget
                            prefill_budget += self.special_settings["method_params"][
                                "window_size"
                            ]

                        past_key_values.apply_special(
                            is_prefill=True, capacity_override=prefill_budget
                        )

                    chunk = prefilling_input_ids[:, index : index + size]
                    # If (1) we do patching and (2) this is not the last chunk, then we need to:
                    # (a) append the last window_size tokens from prefilling_input_ids to the chunk
                    # (b) run the forward pass and let everything be cached
                    # (c) remove the last window_size tokens from the chunk and the past_key_values
                    # This makes sure that caching decisions use the actual last window_size tokens
                    if (
                        "method_params" in self.special_settings
                        and "do_patch" in self.special_settings["method_params"]
                        and self.special_settings["method_params"]["do_patch"]
                        and index + size < prefilling_input_ids.size(1)
                    ):
                        # (a)
                        chunk = torch.cat(
                            [
                                chunk,
                                prefilling_input_ids[
                                    :,
                                    -self.special_settings["method_params"][
                                        "window_size"
                                    ] :,
                                ],
                            ],
                            dim=1,
                        )

                    # print(f"Prefilling {index} -> {index + size} of {prefilling_input_ids.size(1)}")
                    output = self.model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        **extra,
                    )
                    past_key_values = output.past_key_values

                    # (c)
                    if (
                        "method_params" in self.special_settings
                        and "do_patch" in self.special_settings["method_params"]
                        and self.special_settings["method_params"]["do_patch"]
                        and index + size < prefilling_input_ids.size(1)
                    ):
                        past_key_values.drop_last_k_tokens(
                            self.special_settings["method_params"]["window_size"]
                        )

                if (
                    self.special_settings.get("is_pyramid_snapkv", False)
                    and past_key_values is not None
                ):
                    # We are no longer prefilling
                    past_key_values.reset_special()

                inputs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "past_key_values": past_key_values,
                }
            else:
                if "past_key_values" in self.special_settings:
                    past_key_values_class, past_key_values_config = (
                        self.special_settings["past_key_values"]
                    )
                    past_key_values = past_key_values_class(past_key_values_config)

                if self.special_settings.get("is_pyramid_snapkv", False):
                    # The entire KV budget goes to this chunk
                    if "sparsity" in self.special_settings:
                        capacity_override = int(
                            (1 - self.special_settings["sparsity"])
                            * inputs.input_ids.size(1)
                        )
                    else:
                        capacity_override = self.special_settings[
                            "total_prefill_budget"
                        ]

                    # Round up to nearest multiple of 64 and take max with the local_window_size
                    capacity_override = max(
                        capacity_override, self.special_settings["local_window_size"]
                    )
                    capacity_override = ((capacity_override + 63) // 64) * 64
                    past_key_values.apply_special(
                        is_prefill=True, capacity_override=capacity_override
                    )
                    extra["past_key_values"] = past_key_values

                # We don't need to worry about any patching here - the last few tokens are included in the prefill
                prefill = self.model.model(
                    input_ids=inputs.input_ids[..., :-1],
                    attention_mask=inputs.attention_mask[..., :-1],
                    **extra,
                )
                past_key_values = prefill.past_key_values

                if (
                    self.special_settings.get("is_pyramid_snapkv", False)
                    and past_key_values is not None
                ):
                    # We are no longer prefilling
                    past_key_values.reset_special()

                inputs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "past_key_values": past_key_values,
                }
                if past_key_values is None:
                    self.disable_prefill = True
                    logger.warning(
                        "past key values is None, not able to prefill with KVs, disabling..."
                    )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

        text = self.tokenizer.decode(
            outputs["sequences"][0, input_len:], skip_special_tokens=True
        )
        save_prompt = (
            self.tokenizer.decode(inputs["input_ids"][0][:500])
            + " <skip> "
            + self.tokenizer.decode(inputs["input_ids"][0][-500:])
        )

        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs["sequences"].size(1) - input_len,
            "input_text": save_prompt,
        }


class VLLMModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        from vllm import LLM

        # at the time of testing: note that the max model length is derived from the config file, and if max_length is larger than that length, there will be an error. it appears that vllm does not support positional extrapolation
        # there are some work arounds to this, but it may give unexpected results.
        self.model = LLM(
            model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            trust_remote_code=True,
            # enforce_eager=True,
        )
        self.tokenizer = self.model.get_tokenizer()

    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item,
            data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            use_chat_template=self.use_chat_template,
        )

    def generate(self, inputs=None, prompt=None, **kwargs):
        from vllm import SamplingParams, TokensPrompt

        if inputs is None:
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=self.max_length - self.generation_max_length,
                truncation=True,
                padding=True,
            )

        self.sampling_params = SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            max_tokens=self.generation_max_length,
        )

        outputs = self.model.generate(
            prompts=TokensPrompt(prompt_token_ids=inputs["input_ids"][0].tolist()),
            sampling_params=self.sampling_params,
            **kwargs,
        )[0]
        save_prompt = (
            self.tokenizer.decode(inputs["input_ids"][0][:500])
            + " <skip> "
            + self.tokenizer.decode(inputs["input_ids"][0][-500:])
        )
        return {
            "output": outputs.outputs[0].text,
            "input_len": len(outputs.prompt_token_ids),
            "output_len": len(outputs.outputs[0].token_ids),
            "input_text": save_prompt,
        }


def load_LLM(args):
    kwargs = {}
    if "gpt" in args.model_name_or_path:
        model_cls = OpenAIModel
    elif "claude" in args.model_name_or_path:
        model_cls = AnthropicModel
    elif "gemini" in args.model_name_or_path:
        model_cls = GeminiModel
    elif "togetherapi" in args.model_name_or_path:
        model_cls = TogetherModel
    elif args.use_vllm:
        model_cls = VLLMModel
    else:
        model_cls = HFModel
        if args.no_torch_compile:
            kwargs["torch_compile"] = False
        if args.no_bf16:
            kwargs["torch_dtype"] = torch.float32
        if args.rope_theta is not None:
            kwargs["rope_theta"] = args.rope_theta
        print(f"=========args=========={args}")
        if args.elasticattn is not None:
            kwargs["elasticattn"] = args.elasticattn
            kwargs["duoattn_chunk_prefilling"] = args.duoattn_chunk_prefilling

        if args.duoattn is not None:
            kwargs["duoattn"] = args.duoattn
            kwargs["duoattn_sparsity"] = args.duoattn_sparsity
            kwargs["duoattn_sink"] = args.duoattn_sink
            kwargs["duoattn_sliding"] = args.duoattn_sliding
            kwargs["duoattn_chunk_prefilling"] = args.duoattn_chunk_prefilling
            kwargs["duoattn_flipping"] = args.duoattn_flipping
            kwargs["attn_implementation"] = "eager"

        if hasattr(args, "fastprefill_metric") and args.fastprefill_metric is not None:
            kwargs["fastprefill_metric"] = args.fastprefill_metric
            kwargs["fastprefill_threshold"] = args.fastprefill_threshold
            kwargs["fastprefill_print_detail"] = args.fastprefill_print_detail
            kwargs["fastprefill_stride"] = args.fastprefill_stride
            kwargs["duoattn_chunk_prefilling"] = args.duoattn_chunk_prefilling

        if args.minference is not None:
            kwargs["minference"] = args.minference
            kwargs["minference_model_name"] = args.minference_model_name
            kwargs["minference_chunk_prefilling"] = args.minference_chunk_prefilling
            kwargs["minference_window_size"] = args.minference_window_size
            kwargs["minference_max_capacity_prompt"] = (
                args.minference_max_capacity_prompt
            )
            kwargs["minference_sparsity"] = args.minference_sparsity
            kwargs["minference_chunking_patch"] = args.minference_chunking_patch
            kwargs["minference_grouped_eviction"] = args.minference_grouped_eviction
            kwargs["minference_compress_group_kvs"] = args.minference_compress_group_kvs

    model = model_cls(
        args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.input_max_length,
        generation_max_length=args.generation_max_length,
        generation_min_length=args.generation_min_length,
        do_sample=args.do_sample,
        stop_newline=args.stop_newline,
        use_chat_template=args.use_chat_template,
        **kwargs,
    )

    return model
