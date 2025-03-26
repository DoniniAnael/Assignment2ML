import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# import os

# Set GPU device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model paths
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_path = "./results/checkpoint-2019/"  # Path to your saved LoRA adapters


# HuggingFace login if needed
from huggingface_hub import login
login(token="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")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model with 8-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    
    torch_dtype=torch.float16
)

# Load the fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, model_path)
print("Model loaded successfully!")

model = model.merge_and_unload()
model.eval()

def generate_text(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_and_detect_spam(task_type="generate", input_message=None):
    if task_type == "generate":
        prompt = "Generate a spam message"
    elif task_type == "detect":
        if not input_message:
            raise ValueError("input_message is required for spam detection")
        prompt = f"Classify the following message as spam or not:\n\n{input_message}\n\nAnswer:"
    else:
        raise ValueError("task_type should be either generate or detect")

    result = generate_text(prompt)
    return result

# Test the model
print("==== Testing Spam Generation ====")
generated_spam = generate_and_detect_spam(task_type="generate")
print("Generated Spam:", generated_spam)

print("\n==== Testing Spam Detection ====")
# Test with the generated spam
detected_spam = generate_and_detect_spam(task_type="detect", input_message=generated_spam)
print("Detection Result:", detected_spam)

# Test with a custom message
custom_message = "Congratulations! You've won a free iPhone. Click here to claim your prize: http://bit.ly/claim-prize"
print("\n==== Testing Custom Message ====")
print("Message:", custom_message)
detected_custom = generate_and_detect_spam(task_type="detect", input_message=custom_message)
print("Detection Result:", detected_custom)