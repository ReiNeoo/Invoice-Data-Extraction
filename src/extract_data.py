import json
import ollama


class ExtractData:
    def __init__(self):
        pass

    def get_information(self, raw_text):
        query = f"""Extract the following fields from invoice text and output ONLY a JSON object with these keys:

                    totalAmount: The final amount customer must pay (including taxes)
                    
                    Look for: "Vergiler Dahil Toplam Tutar", "Ödenecek Tutar", or similar terms
                    
                    
                    invoiceNumber: The invoice's unique identifier
                    
                    Look for: "E-Arşiv No", "E-ARŞİV NO", "Fatura NO", or similar
                    Format: 1-3 letters followed by 13-15 digits (16 characters total, no spaces)
                    Located in the invoice information section
                    
                    Cluster: It must be same cluster with invoice informations.
                    

                    sellerRegistrationNumber: The seller's tax or registration ID
                    
                    Look for: "VKN", "TCKN", "vergi numarası", or similar
                    Located in the seller information section
                    
                    Cluster: It must be same cluster with sellers information.
                    
                    Output format example:
                    json{{
                      "totalAmount": "123.45 TL",
                      "invoiceNumber": "ABC1234567890123",
                      "sellerRegistrationNumber": "1234567890"
                    }}
                    Return ONLY the JSON object without any additional text, explanations, or formatting. 
                    
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

    def _jsonify_response(self, response):
        _response = response.strip("```json\n").strip("```")
        return json.dumps(_response)
