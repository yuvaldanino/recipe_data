  # This script is used to push the model and tokenizer to the hub

  # Import the necessary libraries
  #   pip install transformers


  
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("tinyllama_model")
   tokenizer = AutoTokenizer.from_pretrained("tinyllama_model")

   model.push_to_hub("ydanino/tinyllama-recipe-finetuned")
   tokenizer.push_to_hub("ydanino/tinyllama-recipe-finetuned")