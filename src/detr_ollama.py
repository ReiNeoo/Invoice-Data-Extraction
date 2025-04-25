# import PIL.Image
# import PIL
# import io
# import os
# import tempfile
# import ollama
# import torch
# import json
# import fitz


# from transformers import AutoImageProcessor
# from transformers import AutoModel, AutoTokenizer
# from transformers.models.detr import DetrForSegmentation


# class ExtractData:
#     def __init__(self):
#         pass

#     def get_information(self, raw_text):
#         print(raw_text)
#         query = f"""Analyze the following text extracted from an invoice and format it into a JSON structure. Return ONLY the JSON output without any additional text or explanation.
#                     If any field is not explicitly labeled, infer its value based on context or common synonyms.
#                     Fields to extract:
#                     1. Total Amount (with tax, what the customer must pay) — may appear as "Vergiler Dahil Toplam Tutar", "Ödenecek Tutar", or similar.
#                     2. Invoice Log Number — labeled as "E-Arşiv No", "E-ARŞİV NO", or similar. It must be a code
#                     3. Seller's Registration Number — labeled as , "VKN", "TCKN", or similar. It must be a code
#                     Return result in this exact JSON format:
#                     {{
#                     "toplam tutar": "",
#                     "fatura no": "",
#                     "satici no": ""
#                     }}

#                     text: {raw_text}
#                     """
#         response = ollama.chat(
#             model='gemma3:4b',
#             messages=[
#                 {'role': 'system', 'content': 'you are an invoice extraction expert.'},
#                 {
#                     'role': 'user',
#                     'content': query,
#                 }
#             ]
#         )
#         return response['message']['content']


# class ExtractDataFromFile(ExtractData):
#     def __init__(self):
#         super().__init__(),
#         pass

#     def get_pages(self, pdf_file: str) -> None:
#         doc = fitz.open(pdf_file)
#         result = [page.get_text() for page in doc]
#         return result

#     def _jsonify_response(self, response):
#         _response = response.strip("```json\n").strip("```")
# #         return json.dumps(_response)


# class ExtractDataFromImage(ExtractData):
#     def __init__(self):
#         super().__init__()
#         self.img_proc = AutoImageProcessor.from_pretrained(
#             "cmarkea/detr-layout-detection")
#         self.layout_model = DetrForSegmentation.from_pretrained(
#             "cmarkea/detr-layout-detection")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             'ucaslcl/GOT-OCR2_0', trust_remote_code=True)
#         self.ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True,
#                                                    device_map='cuda', use_safetensors=True, pad_token_id=self.tokenizer.eos_token_id)
#         self.ocr_model = self.ocr_model.eval().cuda()

#     def _get_bbox(self, image_path):
#         image = PIL.Image.open(image_path)
#         target_size = [image.size[::-1]]  # (H, W)
#         with torch.inference_mode():
#             inputs = self.img_proc(image, return_tensors='pt')
#             outputs = self.layout_model(**inputs)

#         threshold = 0.85
#         results = self.img_proc.post_process_object_detection(
#             outputs, threshold=threshold, target_sizes=target_size)[0]

#         return results["boxes"], results["labels"]

#     def _crop_image(self, image_path: str, bboxes, labels) -> list:
#         cropped_images = []
#         image = PIL.Image.open(image_path)
#         for bbox, label in zip(bboxes, labels):
#             if self.layout_model.config.id2label[label.item()] != 'Picture':
#                 xmin, ymin, xmax, ymax = bbox.tolist()
#                 cropped_image = image.crop((xmin, ymin, xmax, ymax))
#                 cropped_images.append(cropped_image)
#         return cropped_images

#     def extract_text_with_OCR(self, image_path: str):
#         total_string = ""
#         bboxes, labels = self._get_bbox(image_path)
#         cropped_images = self._crop_image(image_path, bboxes, labels)

#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save each image temporarily
#             for i, piece in enumerate(cropped_images):
#                 temp_path = os.path.join(temp_dir, f"temp_image_{i}.png")
#                 piece.save(temp_path)
#                 res = self.ocr_model.chat(
#                     self.tokenizer, temp_path, ocr_type='ocr')

#                 total_string += res + "\n"
#         return total_string
