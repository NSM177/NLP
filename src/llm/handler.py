"""
Generic Local LLM handler using HuggingFace Transformers.
Model name loaded from .env (LLM_MODEL_NAME), default: Meta-Llama-3-8B-Instruct.
Supports 4-bit quantization, multi-GPU distribution, and KV cache offloading.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from src.config import (
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.llm.base import BaseLLM


class LocalLLM(BaseLLM):
    """
    Generic local LLM running via HuggingFace Transformers.
    Supports any causal LM (Llama, Qwen, Mistral, etc.)
    """

    def __init__(self, model_name: str = None, use_4bit: bool = True, device_map: str = "auto"):
        """
        Khởi tạo lớp LocalLLM.

        Args:
            model_name: Tên mô hình hoặc đường dẫn (nếu None thì dùng LLM_MODEL_NAME từ config).
            use_4bit: Nếu True, sử dụng lượng tử hóa 4-bit (giảm mạnh VRAM). 
                       Nếu model đã là GPTQ, sẽ tự động tắt use_4bit.
            device_map: Chiến lược phân bổ device, "auto" hoặc "balanced".
        """
        self.model_name = model_name or LLM_MODEL_NAME
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        Nạp mô hình và tokenizer, hỗ trợ 4-bit và multi-GPU, tự động nhận diện GPTQ.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Tự động phát hiện model GPTQ (đã được lượng tử hóa sẵn)
            is_gptq = "GPTQ" in self.model_name
            if is_gptq:
                print(f"Detected GPTQ model: {self.model_name}. Disabling additional quantization.")
                bnb_config = None
                self.use_4bit = False  # override
            else:
                # Cấu hình lượng tử hóa 4-bit dynamic
                if self.use_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    print(f"Loading {self.model_name} with 4-bit quantization...")
                else:
                    bnb_config = None
                    print(f"Loading {self.model_name} without quantization...")

            # Tự động phát hiện số GPU và giới hạn bộ nhớ
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                # Giảm max_memory mỗi GPU để chừa chỗ cho SLM
                max_memory = {i: "10GiB" for i in range(num_gpus)}
                max_memory["cpu"] = "20GiB"  # Offload sang RAM nếu cần
                print(f"Using {num_gpus} GPUs with max_memory {max_memory}")
            else:
                max_memory = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16 if not self.use_4bit and not is_gptq else "auto",
                device_map=self.device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Loaded LLM: {self.model_name} on device map: {self.model.hf_device_map}")
        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")

    def generate_text(
        self, prompt: str, max_output_tokens: int = LLM_MAX_NEW_TOKENS
    ) -> str:
        """Sinh văn bản từ prompt với KV cache offloading để tiết kiệm VRAM."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model is not initialized.")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        # Cấu hình generation với offload KV cache để giảm VRAM
        gen_config = GenerationConfig(
            max_new_tokens=max(1, int(max_output_tokens)),
            do_sample=False,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            cache_implementation="offloaded",  # Quan trọng: offload KV cache sang CPU
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        gen_only_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.tokenizer.batch_decode(
            gen_only_ids, skip_special_tokens=True
        )[0]
        
        torch.cuda.empty_cache()
        return response.strip()


# ============================================================
# Singleton Accessor
# ============================================================
_current_llm = None


def get_llm(model_name: str = None, use_4bit: bool = True, device_map: str = "auto") -> BaseLLM:
    """
    Lấy hoặc tạo mới singleton Global LLM.
    """
    global _current_llm
    if _current_llm is None:
        _current_llm = LocalLLM(model_name, use_4bit=use_4bit, device_map=device_map)
    return _current_llm