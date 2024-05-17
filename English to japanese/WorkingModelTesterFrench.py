from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Set up configurations for encoder and decoder with specific modifications for the decoder
encoder_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
decoder_config = BertConfig.from_pretrained("bert-base-multilingual-cased", decoder_start_token_id=tokenizer.cls_token_id)

# Create an EncoderDecoderConfig
enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

# Load the model
model_path = 'C://Users//romed//Desktop//English to japanese//bert_translation_model_french_edition.pth'
model = EncoderDecoderModel.from_pretrained(model_path, config=enc_dec_config)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def translate(sentence, model, tokenizer):
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    #print("Input IDs:", inputs['input_ids'])

    # Using no_repeat_ngram_size to avoid repetition, adjust according to your needs
    output_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        decoder_start_token_id=tokenizer.bos_token_id,
        max_length=128,
        no_repeat_ngram_size=2
    )
    #print("Output IDs:", output_ids)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation

# Example usage
sentence = "This is an example sentence."
translation = translate(sentence, model, tokenizer)
print("Translated Sentence:", translation)
# Example usage
sentence = "We will talk about challenges next"
translation = translate(sentence, model, tokenizer)
print("Translated Sentence:", translation)

# Example usage
sentence = "Hi how was your day?"
translation = translate(sentence, model, tokenizer)
print("Translated Sentence:", translation)


