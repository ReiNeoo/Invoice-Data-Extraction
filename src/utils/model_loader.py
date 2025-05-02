class ModelLoader:
    @staticmethod
    def load_yolo_model(model_path):
        from doclayout_yolo import YOLOv10
        return YOLOv10(model_path)

    @staticmethod
    def load_ocr_model():
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return model, tokenizer
