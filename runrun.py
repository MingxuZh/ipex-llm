import os
import argparse
import json
import re
import time
import gc
import torch
from datasets import load_dataset
from pathlib import Path
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaTokenizer,
)


MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs="?", default="EleutherAI/gpt-j-6b")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--device", default="cpu", type=str, help="cpu")
parser.add_argument(
    "--dtype", default="bfloat16", type=str, help="float32 or bfloat16 or int8"
)
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument(
    "--batch-size", default=1, type=int, help="For accuracy measurement only."
)
parser.add_argument(
    "--save-accuracy-path", default=None, help="Save accuracy results path."
)
parser.add_argument(
    "--ipex", action="store_true", help="use intel extension for pytorch."
)
parser.add_argument(
    "--jit", action="store_true", help="convert model to torchscript mode."
)
parser.add_argument("--int8-bf16-mixed", action="store_true", help="int8 mixed bf16")
parser.add_argument("--quantized-model-path", default="./saved_result/best_model.pt")
parser.add_argument(
    "--tasks",
    nargs="+",
    default=[
        "lambada_openai",
    ],
    type=str,
    help="tasks list for accuracy validation, only enabled lambada_openai and lambada_standard at present",
)
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)

args = parser.parse_args()

# disable
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

# beam search = 4
num_beams = 1 if args.greedy else 4
generate_kwargs = dict(
    do_sample=False,
    temperature=0.9,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
)

# load model
config = AutoConfig.from_pretrained(
    args.model_id, torchscript=args.jit, trust_remote_code=True
)
if args.kv_cache_dtype != "None":
    args.kv_cache_dtype = getattr(torch, args.kv_cache_dtype)
    print("kv_cache_dtype:", args.kv_cache_dtype)
    config.kv_cache_dtype = args.kv_cache_dtype

if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

if args.benchmark and args.jit and not args.ipex_weight_only_quantization:
    try:
        with ipex._IPEXOnDevice(dtype=torch.float, device="meta"):
            user_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )
    except:
        user_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
print("Data type of the model:", user_model.dtype)

# to channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()
# calling _optimize_transformers for int8 path
if args.ipex_weight_only_quantization:
    user_model.config.weight_only_quantization = True
user_model = ipex._optimize_transformers(
    user_model.eval(), dtype=torch.int8, inplace=True
)

beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()
global_past_key_value = [
    (
        torch.zeros([1, 1, 1, 1]).contiguous()
        if args.kv_cache_dtype == "None"
        else torch.zeros([1, 1, 1, 1], dtype=args.kv_cache_dtype).contiguous(),
        torch.zeros([1, 1, 1, 1]).contiguous()
        if args.kv_cache_dtype == "None"
        else torch.zeros([1, 1, 1, 1], dtype=args.kv_cache_dtype).contiguous(),
        beam_idx_tmp,
        torch.zeros(1, dtype=torch.long).contiguous(),
    )
    for i in range(user_model.config.num_hidden_layers)
]

# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

if args.lambada:

    class Evaluator:
        def __init__(
            self, dataset, tokenizer, args, batch_size=8, pad_val=1, pad_max=196
        ):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.pad_val = pad_val
            self.pad_max = pad_max
            self.args = args

            # tokenize the dataset
            self.dataset = self.dataset.map(self.tokenize_function, batched=True)
            self.dataset.set_format(type="torch", columns=["input_ids"])

        @torch.no_grad()
        def tokenize_function(self, examples):
            example = self.tokenizer(examples["text"])
            return example

        @torch.no_grad()
        def collate_batch(self, batch):
            input_ids_padded = []
            last_ind = []
            past_key_values = []
            attention_mask_padded = []

            for text in batch:
                input_ids = text["input_ids"]
                pad_len = self.pad_max - input_ids.shape[0]
                last_ind.append(input_ids.shape[0] - 1)

                attention_mask = torch.ones(len(input_ids))

                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
                input_ids_padded.append(input_ids)
                attention_mask = pad(attention_mask, (0, pad_len), value=0)
                attention_mask_padded.append(attention_mask)

                # dummy past key value
                beam_idx_tmp = torch.zeros(
                    (2048, int(self.args.batch_size * num_beams)), dtype=torch.long
                ).contiguous()
                past_key_value = [
                    (
                        torch.zeros([1, 16, 1, 256]).contiguous(),
                        torch.zeros([1, 16, 1, 256]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, dtype=torch.long).contiguous(),
                    )
                    for i in range(user_model.config.num_hidden_layers)
                ]

            return (
                (
                    torch.vstack(input_ids_padded),
                    torch.vstack(attention_mask_padded),
                    tuple(past_key_value),
                ),
                torch.tensor(last_ind),
            )

        @torch.no_grad()
        def evaluate(self, model):
            model.eval()
            # The task is to predict the last word of the input.
            total, hit = 0, 0
            latency = 0
            test_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_batch,
            )
            for i, (
                (input_ids, attention_mask, past_key_values),
                last_ind,
            ) in enumerate(test_dataloader):
                label = input_ids[torch.arange(len(last_ind)), last_ind]
                input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
                pad_len = self.pad_max - last_ind - 1
                start = time.time()
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                latency += time.time() - start
                last_token_logits = outputs[0][
                    torch.arange(len(last_ind)), -2 - pad_len, :
                ]
                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                if i % 50 == 0:
                    print(hit / total, flush=True)
                    print("Processed minibatch:", i, flush=True)

            acc = hit / total
            print(acc)
            lantecy = latency / len(self.dataset)
            return acc, lantecy

    full_dataset = load_dataset("lambada")
    dataset = full_dataset["validation"]
    calib_dataset = full_dataset["train"]
    evaluator = Evaluator(dataset, tokenizer, args, batch_size=args.batch_size)
    calib_evaluator = Evaluator(
        calib_dataset, tokenizer, args, batch_size=args.batch_size
    )

    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )


def calib_func(prepared_model):
    for i, (
        (input_ids, attention_mask, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        if i == 8:
            break
        prepared_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )


def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


quant_model = user_model

if args.ipex_smooth_quant:
    from neural_compressor import PostTrainingQuantConfig, quantization

    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["kl"],
            },
        },
    }

    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": "auto"}}

    recipes["smooth_quant_args"]["folding"] = True

    print("smooth_quant_args:", recipes)
    conf = PostTrainingQuantConfig(
        backend="ipex",
        excluded_precisions=excluded_precisions,
        op_type_dict=op_type_dict,
        recipes=recipes,
    )

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
    )

    q_model.save(args.output_dir)
    quant_model = q_model

if args.accuracy_only:
    import lm_eval
    from lm_eval import tasks, evaluator, models
    from lm_eval.base import BaseLM
    from typing import Union, List, Optional
    from transformers import BatchEncoding

    TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

    class HuggingFaceModel(BaseLM):
        _DEFAULT_MAX_LENGTH = 2048

        def __init__(
            self,
            device="cpu",
            model_id="",
            with_ipex=True,
            with_jit=True,
            with_greedy=False,
            batch_size=1,
            max_length=None,
            dtype: Optional[Union[str, torch.dtype]] = "auto",
            config=None,
        ):
            super().__init__()

            self._device = device
            self._batch_size = batch_size
            self._with_jit = with_jit
            self._with_ipex = with_ipex
            self._with_greedy = with_greedy
            self._max_length = max_length
            self._dtype = dtype

            if dtype == "float16":
                load_dtype = torch.half
                infer_dtype = torch.half
            elif dtype == "bfloat16":
                load_dtype = torch.bfloat16
                infer_dtype = torch.bfloat16
            elif dtype == "int8":
                load_dtype = torch.float32
                infer_dtype = torch.int8
            elif dtype == "float32":
                load_dtype = torch.float32
                infer_dtype = torch.float32

            amp_enabled = True if dtype != "float32" else False
            amp_dtype = getattr(torch, dtype)

            model_type = next(
                (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
            )
            model_class = MODEL_CLASSES[model_type]
            self.tokenizer = model_class[1].from_pretrained(
                model_id, trust_remote_code=True
            )
            if config is None:
                self.config = AutoConfig.from_pretrained(
                    model_id, torchscript=with_jit, trust_remote_code=True
                )
            else:
                self.config = AutoConfig.from_pretrained(
                    config, torchscript=with_jit, trust_remote_code=True
                )

            self.model = model_class[0].from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                config=self.config,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )

            self.model = self.model.eval()

            if with_ipex:
                self.model = ipex._optimize_transformers(
                    self.model.eval(), dtype=infer_dtype, inplace=True
                )

            self.base_model = self.model

            self.iter = 0
            self.num_beams = 1 if with_greedy else 4
            self.tp_number = 1

        def _model_call(
            self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
        ) -> TokenSequence:
            _attention_mask = []
            _position_ids = []

            if self._with_jit:
                for text in inputs:
                    input_ids = text.to(self._device)
                    input_bs = inputs.shape[0] * self.num_beams
                    if (
                        re.search("GPTJ", self.base_model.config.architectures[0])
                        or re.search(
                            "codegen",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                        or re.search(
                            "gptbigcode",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.n_head
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.n_embd
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.n_head
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.n_embd
                                                / self.base_model.config.n_head
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.n_layer)
                            ]
                        )
                    elif re.search(
                        "llama", self.base_model.config.architectures[0], re.IGNORECASE
                    ) or re.search(
                        "baichuan",
                        self.base_model.config.architectures[0],
                        re.IGNORECASE,
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "gptneox",
                        self.base_model.config.architectures[0],
                        re.IGNORECASE,
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "bloom", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "OPT", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "falcon", self.base_model.config.architectures[0], re.IGNORECASE
                    ) or re.search(
                        "rw", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        num_hidden_layers = (
                            self.base_model.config.num_hidden_layers
                            if hasattr(self.base_model.config, "num_hidden_layers")
                            else self.base_model.config.n_layer
                        )
                        num_attention_heads = (
                            self.base_model.config.num_attention_heads
                            if hasattr(self.base_model.config, "num_attention_heads")
                            else self.base_model.config.n_head
                        )
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(num_attention_heads / self.tp_number),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(num_attention_heads / self.tp_number),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(num_hidden_layers)
                            ]
                        )
                    elif re.search(
                        "chatglm",
                        self.base_model.config.architectures[0],
                        re.IGNORECASE,
                    ):
                        beam_idx_tmp = torch.zeros(
                            (2048, int(input_bs)), dtype=torch.long
                        ).contiguous()
                        past_key_values = tuple(
                            [
                                (
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    torch.zeros(
                                        [
                                            1,
                                            int(
                                                self.base_model.config.num_attention_heads
                                                / self.tp_number
                                            ),
                                            1,
                                            int(
                                                self.base_model.config.hidden_size
                                                / self.base_model.config.num_attention_heads
                                            ),
                                        ]
                                    ).contiguous(),
                                    beam_idx_tmp,
                                    torch.zeros(1, dtype=torch.long).contiguous(),
                                )
                                for i in range(self.base_model.config.num_layers)
                            ]
                        )

                    position_ids = torch.arange(len(input_ids))
                    attention_mask = torch.ones(len(input_ids))

                    _attention_mask.append(attention_mask)
                    _position_ids.append(position_ids)

                attention_mask_batched = torch.stack(_attention_mask)
                position_ids_batched = torch.stack(_position_ids)

            if self._with_jit and self.iter == 0:
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._dtype != "int8":
                        if (
                            re.search(
                                "bloom",
                                self.base_model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            or re.search(
                                "OPT",
                                self.base_model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            or re.search(
                                "falcon",
                                self.base_model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            or re.search(
                                "rw",
                                self.base_model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            or re.search(
                                "baichuan",
                                self.base_model.config.architectures[0],
                                re.IGNORECASE,
                            )
                        ):
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "past_key_values": past_key_values,
                            }
                        else:
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "position_ids": position_ids_batched,
                                "past_key_values": past_key_values,
                            }
                        self.model = torch.jit.trace(
                            self.model.eval(),
                            example_kwarg_inputs=example_dict,
                            strict=False,
                            check_trace=False,
                        )
                        self.model = torch.jit.freeze(self.model.eval())
                    else:
                        self.model=q_model
                        self.model = torch.jit.freeze(self.model.eval())

                    if (
                        re.search(
                            "bloom",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                        or re.search(
                            "OPT",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                        or re.search(
                            "falcon",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                        or re.search(
                            "rw", self.base_model.config.architectures[0], re.IGNORECASE
                        )
                        or re.search(
                            "baichuan",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        )
                    ):
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )

                self.iter = self.iter + 1

            if (
                re.search(
                    "bloom", self.base_model.config.architectures[0], re.IGNORECASE
                )
                or re.search(
                    "OPT", self.base_model.config.architectures[0], re.IGNORECASE
                )
                or re.search(
                    "falcon", self.base_model.config.architectures[0], re.IGNORECASE
                )
                or re.search(
                    "rw", self.base_model.config.architectures[0], re.IGNORECASE
                )
                or re.search(
                    "baichuan", self.base_model.config.architectures[0], re.IGNORECASE
                )
            ):
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._with_jit:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        output = self.base_model(
                            inputs,
                        )
            else:
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._with_jit:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                    else:
                        output = self.base_model(
                            inputs,
                        )

            if isinstance(output, tuple):
                return output[0]

            return output["logits"]

        @property
        def eot_token_id(self):
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            if self._max_length:  # if max length manually set, return it
                return self._max_length
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self.config, attr):
                    return getattr(self.config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length

            return self._DEFAULT_MAX_LENGTH

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            # TODO: fix multi-gpu
            return self._batch_size  # * gpus

        @property
        def device(self):
            # TODO: fix multi-gpu
            return self._device

        def tok_encode(self, string: str):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _model_generate(self, context, max_length, eos_token_id):
            generation_kwargs = {"do_sample": False, "max_length": max_length}
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
                generation_kwargs[
                    "pad_token_id"
                ] = eos_token_id  # setting eos_token_id as pad token
            return self.model.generate(context, **generation_kwargs)

    task_dict = lm_eval.tasks.get_task_dict(args.tasks)

    hfmodel = HuggingFaceModel(
        model_id=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=args.ipex,
        with_jit=args.jit,
        dtype=args.dtype,
        config=args.config_file,
    )

    results = evaluator.evaluate(
        hfmodel,
        task_dict,
        #        bootstrap_iters=1000,
        #        limit=100
    )

    print(evaluator.make_table(results))