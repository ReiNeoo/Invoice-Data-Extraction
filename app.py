import gradio as gr
import os

from src.extract_data_image import ExtractDataFromImageYOLO
from src.extract_data_file import ExtractDataFromFile


class ProcessFiles:
    def __init__(self):
        self.extractor_image = ExtractDataFromImageYOLO()
        self.extractor_file = ExtractDataFromFile()

    def extract_image(self, image_file):
        text = self.extractor_image.extract_text_with_OCR(image_file)
        information = self.extractor_image.get_information(text)

        return information

    # def extract_image(self, image_file):
    #     text = self.extractor_image.extract_text_with_OCR(image_file)
    #     information = self.extractor_image.get_information(text)

    #     return information

    def extract_file(self, pdf_file):
        text = self.extractor_file.get_pages(pdf_file)
        information = self.extractor_file.get_information(text)

        return information

    def process_file(self, file):
        file_path = file.name
        file_ext = os.path.splitext(file_path)[1].lower()

        result = ""

        try:
            if file_ext == ".pdf":
                information = self.extract_file(file_path)
                result += information

            elif file_ext in [".png", ".jpg", ".jpeg"]:
                information = self.extract_image(file_path)
                result += information

        except Exception as e:
            result = f"Dosya işlenirken bir hata oluştu: {str(e)}"
        return result


# Gradio arayüzü oluşturuluyor
with gr.Blocks(title="Fatura Bilgi Çıkartma Uygulaması") as app:
    processor = ProcessFiles()
    gr.Markdown("# Dosya İşleme Uygulaması")
    gr.Markdown(
        "PDF veya görüntü dosyalarını (PNG, JPG, vb.) yükleyin ve analiz sonucunu görün.")

    with gr.Row():
        file_input = gr.File(label="Dosya Yükleyin")

    with gr.Row():
        process_btn = gr.Button("Dosyayı İşle")

    with gr.Row():
        output = gr.TextArea(
            label="Sonuç", placeholder="Sonuçlar burada görüntülenecek...", lines=15)

    # Buton tıklandığında işlemi başlatıyoruz
    process_btn.click(fn=processor.process_file,
                      inputs=file_input, outputs=output)

    # Dosya yüklendiğinde otomatik işleme (isteğe bağlı)
    file_input.change(fn=processor.process_file,
                      inputs=file_input, outputs=output)

# Uygulamayı başlatıyoruz
if __name__ == "__main__":
    app.launch(share=True)
