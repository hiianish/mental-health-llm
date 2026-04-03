from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="E:\projects\mental_health_lora_adapter\final_adapter", 
    max_seq_length=1024,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


inputs = tokenizer(
    "I feel anxious and can't focus on my studies",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))