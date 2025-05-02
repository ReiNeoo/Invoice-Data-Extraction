import PIL.Image
import os
import tempfile
from io import BytesIO

from transformers import AutoImageProcessor
from transformers import AutoModel, AutoTokenizer
from transformers.models.detr import DetrForSegmentation
from doclayout_yolo import YOLOv10

from src.extract_data import ExtractData


class ExtractDataFromImageYOLO(ExtractData):
    def __init__(self):
        super().__init__()
        self.model = YOLOv10(
            "models/doclayout_yolo_docstructbench_imgsz1024.pt")
        self.device = "cuda:0"
        self.conf = 0.1
        self.imgsize = 1024

        self.tokenizer = AutoTokenizer.from_pretrained(
            'ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        self.ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True,
                                                   device_map='cuda', use_safetensors=True, pad_token_id=self.tokenizer.eos_token_id)
        self.ocr_model = self.ocr_model.eval().cuda()

    def _get_yolo_attr(self, boxes):
        bboxes = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        class_names = self.model.names if hasattr(
            self.model, 'names') else None

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
        results = self.model.predict(
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

    def extract_text_with_OCR_captcha(self, response):
        image = PIL.Image.open(BytesIO(response))
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each image temporarily
            temp_path = os.path.join(temp_dir, f"temp_image_captcha.png")
            image.save(temp_path)
            res = self.ocr_model.chat(
                self.tokenizer, temp_path, ocr_type='ocr'
            )
            return res

    def extract_text_with_OCR(self, image_path: str):
        total_string = ""
        bboxes, labels = self._get_bbox_with_yolo(image_path)
        cropped_images = self._crop_yolo_image(image_path, bboxes, labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each image temporarily
            for i, piece in enumerate(cropped_images):
                temp_path = os.path.join(temp_dir, f"temp_image_{i}.png")
                piece.save(temp_path)
                res = self.ocr_model.chat(
                    self.tokenizer, temp_path, ocr_type='ocr')

                total_string += f"Text Cluster_{i}: {res}" + "\n\n"
        return total_string
