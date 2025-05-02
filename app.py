import gradio as gr
import os

from _src.extractors.xml_extractor import extract_data_from_xml
from _src.extractors.image_extractor import ImageExtractor
from _src.extractors.pdf_extractor import PDFExtractor


class ProcessFiles:
    def __init__(self):
        self.image_extractor = ImageExtractor()
        self.file_extractor = PDFExtractor()

    def process_file(self, file):
        file_path = file.name
        file_ext = os.path.splitext(file_path)[1].lower()

        result = ""

        try:
            if file_ext == ".pdf":
                information = self.file_extractor.extract(file_path)
                result += information

            elif file_ext in [".png", ".jpg", ".jpeg"]:
                information = self.image_extractor.extract(file_path)
                result += information

            elif file_ext == ".xml":
                information = extract_data_from_xml(file_path)
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
