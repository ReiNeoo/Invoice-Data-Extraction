import ollama
from _src.core.base import DataExtractor
from _src.utils.model_loader import ModelLoader
import PIL
import os
import tempfile


class ImageExtractor(DataExtractor):
    def __init__(self):
        self.model_loader = ModelLoader()
        self.yolo_model = self.model_loader.load_yolo_model(
            "models/doclayout_yolo_docstructbench_imgsz1024.pt")
        self.ocr_model, self.ocr_tokenizer = self.model_loader.load_ocr_model()
        self.ocr_model = self.ocr_model.eval().cuda()

        self.device = "cuda:0"
        self.conf = 0.1
        self.imgsize = 1024

    def extract(self, image_path):
        text = self.extract_text_with_OCR(image_path)
        return self.get_information(text)

    def _get_yolo_attr(self, boxes):
        bboxes = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        class_names = self.yolo_model.names if hasattr(
            self.yolo_model, 'names') else None

        _bboxes = []
        _labels = []
        for box in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[box]
            class_idx = int(classes[box])
            label = class_names[class_idx] if class_names else f"Class {class_idx}"

            _bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            _labels.append(label)

        return _bboxes, _labels

    def _get_bbox_with_yolo(self, image_path):
        results = self.yolo_model.predict(
            image_path,
            imgsz=self.imgsize,
            conf=self.conf,
            device=self.device
        )
        boxes = results[0].boxes
        bboxes, labels = self._get_yolo_attr(boxes)
        return bboxes, labels

    def _crop_yolo_image(self, image_path: str, bboxes, labels) -> list:
        cropped_images = []
        image = PIL.Image.open(image_path)
        for bbox, label in zip(bboxes, labels):
            if label != "abandon":
                xmin, ymin, xmax, ymax = bbox
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_images.append(cropped_image)
        return cropped_images

    def extract_text_with_OCR(self, image_path):
        total_string = ""
        bboxes, labels = self._get_bbox_with_yolo(image_path)
        cropped_images = self._crop_yolo_image(image_path, bboxes, labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each image temporarily
            for i, piece in enumerate(cropped_images):
                temp_path = os.path.join(temp_dir, f"temp_image_{i}.png")
                piece.save(temp_path)
                res = self.ocr_model.chat(
                    self.ocr_tokenizer, temp_path, ocr_type='ocr')

                total_string += f"Text Cluster_{i}: {res}" + "\n\n"
        return total_string

    def get_information(self, raw_text):
        query = f"""First understand all clusters then Extract the following fields, according to given cluster informations, from invoice text and output ONLY a JSON object with these keys:
        
                    totalAmount: The final amount customer must pay (including taxes)

                    Look for: "Vergiler Dahil Toplam Tutar", "Ödenecek Tutar", "Net Alınan Tutar", "Net Tahsilat" or similar terms

                    Cluster: it must be same cluster with payment information.

                    invoiceNumber: The invoice's unique identifier

                    Look for: "E-Arşiv No", "E-ARŞİV NO", "Fatura NO", "Belge NO", or similar. It is Not ETTN number.
                    Format: 1-3 letters followed by 13-15 digits (16 characters total, no spaces)
                    Located in the invoice information section

                    Cluster: It must be same cluster with invoice informations.

                    sellerRegistrationNumber: The seller's tax or registration ID

                    Look for: "VKN", "TCKN", "vergi numarası", or similar
                    Located in the seller information section

                    Cluster: It must be same cluster with sellers information like "DÜZENLEYEN" or Vendor name.

                    Output format example:
                    json{{
                      "totalAmount": "123.45 TL",
                      "invoiceNumber": "ABC1234567890123",
                      "sellerRegistrationNumber": "1234567890"
                    }}
                    Return ONLY the JSON object without any additional text, explanations, or formatting.
                    Ignore clusters that contains Customer information like "Alıcı Bilgileri", Customer name, The bank informations. 
                    Be AWARE of CLUSTERS in order to decide which information is correct. It is the most important part.

                    text: {raw_text}
                    """

        response = ollama.chat(
            model='Llama3.1',
            messages=[
                {'role': 'system', 'content': 'you are an invoice extraction expert.'},
                {
                    'role': 'user',
                    'content': query,
                }
            ]
        )
        return response['message']['content']
