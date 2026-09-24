# Installation and Usage

1. Create three Python environments using the dependency specifications in `app_requirements.txt`, `rag_requirements.txt`, and `vllm_requirements.txt`, respectively.

2. In `start_all.sh`, update the environment names to match the three environments created in Step 1.

3. In the project root directory, create a folder named `model_weights`. Within this folder, create a subfolder named `diagnosis_weights`.

4. Download the diagnostic model weights from the [Hugging Face repository](https://huggingface.co/Sydenial/SpineAgent/weights) and place them in the `model_weights/diagnosis_weights/` directory.

5. Download `Qwen3-Embedding-8B` and place it in the `model_weights/` directory. The final directory structure should be:

```text
model_weights/
├── diagnosis_weights/
└── Qwen3-Embedding-8B/
```

6. Open a terminal in the project root directory and run:

```bash
bash start_all.sh
```
