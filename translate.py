from transformers import MarianMTModel, MarianTokenizer

def translate_to_spanish(texts):
    # Load model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and translate
    translated = []
    for text in texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        translated_ids = model.generate(**inputs)
        
        # Decode the translation
        translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        translated.append(translation)
    
    return translated

def main():
    # Example English sentences
    english_texts = [
        "I love programming",
        "The weather is beautiful today",
        "Let's learn something new"
    ]
    
    # Translate to Spanish
    spanish_translations = translate_to_spanish(english_texts)
    
    # Print results
    for eng, spa in zip(english_texts, spanish_translations):
        print(f"English: {eng}")
        print(f"Spanish: {spa}")
        print("-" * 50)

if __name__ == "__main__":
    main()