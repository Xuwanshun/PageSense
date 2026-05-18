"""
Modal deployment for the fine-tuned Qwen3-VL model.
Serves an OpenAI-compatible /v1/chat/completions endpoint.

USAGE
-----
1. Install Modal and authenticate (one-time):
       pip install modal
       modal setup

2. Deploy:
       modal deploy scripts/modal_vlm.py

   Modal prints a URL like:
       https://<your-workspace>--qwen3-vl-rag-model-fastapi-app.modal.run

3. Set in .env:
       VLM_BASE_URL=https://<your-workspace>--qwen3-vl-rag-model-fastapi-app.modal.run/v1
       VLM_SELF_HOSTED_MODEL=qwen3-vl-rag

4. Tear down when done:
       modal app stop qwen3-vl-rag

The endpoint scales to zero after 5 minutes idle (~$0/hr when idle).
Cost while running: ~$0.60/hr on A10G.
"""

import modal

app = modal.App("qwen3-vl-rag")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=5.0.0",
    "accelerate>=0.30.0",
    "qwen-vl-utils>=0.0.8",
    "Pillow",
    "torchvision",
    "fastapi[standard]",
)

MODEL_ID = "azhuang3/qwen3_vlm_task"


@app.cls(
    gpu="A10G",
    image=image,
    scaledown_window=300,  # scale to zero after 5 min idle
)
@modal.concurrent(max_inputs=4)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @modal.asgi_app()
    def fastapi_app(self):
        import time

        import torch
        from fastapi import FastAPI, Request
        from qwen_vl_utils import process_vision_info

        web = FastAPI()

        @web.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{"id": "qwen3-vl-rag", "object": "model"}],
            }

        @web.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages_in = body.get("messages", [])
            max_tokens = body.get("max_tokens", 512)

            qwen_messages = _convert_messages(messages_in)

            text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(qwen_messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            reply = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": "qwen3-vl-rag",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        def _convert_messages(messages_in: list) -> list:
            """Convert OpenAI-format messages to Qwen3-VL format.

            Passes data URLs directly — qwen_vl_utils handles data:image/...;base64,...
            natively, so no temp files are needed.
            """
            qwen_messages = []
            for msg in messages_in:
                role = msg["role"]
                content = msg["content"]

                if isinstance(content, str):
                    qwen_messages.append({"role": role, "content": content})
                    continue

                qwen_content = []
                for part in content:
                    if part["type"] == "text":
                        qwen_content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        # Pass the URL as-is — works for both data: and https: URLs
                        qwen_content.append({"type": "image", "image": part["image_url"]["url"]})

                qwen_messages.append({"role": role, "content": qwen_content})

            return qwen_messages

        return web
