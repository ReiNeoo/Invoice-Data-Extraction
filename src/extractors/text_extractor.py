from src.core.base import DataExtractor
import ollama


class TextExtractor(DataExtractor):
    def extract(self, text):
        return self.get_information(text)

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
