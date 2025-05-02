import json
import ollama
import os
import tempfile
from PIL import Image
from io import BytesIO


class ExtractData:
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

        # query = f"""
        #         Analyze the following text extracted from a receipt and format it into a JSON structure. Include educated guesses for missing information where appropriate. Return ONLY the JSON output without any additional text or explanation.

        #         For the following fields, if the information is not explicitly provided, make an educated guess based on the context:
        #         - payment details (of transaction)
        #         - merchant information
        #         - date (of transaction)
        #         - hour (of transaction)

        #         Please provide response only in Turkish.

        #         Extracted text: {raw_text}

        #         Return ONLY the JSON object without any additional text, explanations, or formatting.

        #         JSON format must be as follows:
        #         {
        #     "paymentDetails": {
        #         "amount": 270.00,
        #             "paymentMethod": "Credit Card"},
        #         "merchantInformation": {
        #         "name": "Merchant Name",
        #             "city": "City Name",
        #             "address": "Street Address"},
        #         "date": "dd.mm.yyyy",
        #         "time": "hh:mm:ss"
        #         }
        #                         """

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

    def _jsonify_response(self, response):
        _response = response.strip("```json\n").strip("```")
        return json.dumps(_response)
