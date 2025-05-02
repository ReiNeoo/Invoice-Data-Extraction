import xml.etree.ElementTree as ET
import re


def extract_data_from_xml(xml_file_path):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Extract all namespaces used in the document
        namespaces = {}
        for elem in root.iter():
            if '}' in elem.tag:
                uri = elem.tag.split('}')[0].strip('{')
                prefix = elem.tag.split('}')[1].split(
                    ':')[0] if ':' in elem.tag.split('}')[1] else ''
                if prefix and uri:
                    namespaces[prefix] = uri

        # print("Detected namespaces:", namespaces)

        # Initialize result dictionary
        result = {
            "supplier_vkn": "Not found",
            "payment_amount": "Not found",
            "id": "Not found"
        }

        # Method 1: Direct search with full tags
        for elem in root.iter():
            # Check for VKN ID
            if 'ID' in elem.tag and elem.get('schemeID') == 'VKN':
                result["supplier_vkn"] = elem.text

            # Check for payment instruction
            if 'InstructionNote' in elem.tag and elem.text and 'TRY' in elem.text:
                result["payment_amount"] = elem.text

            # Check for document ID
            if 'ID' in elem.tag and elem.text and elem.text.startswith('89820'):
                result["id"] = elem.text

        # Method 2: Search for text patterns as a fallback
        if result["supplier_vkn"] == "Not found":
            for elem in root.iter():
                # Turkish VKN is 10 digits
                if elem.text and re.match(r'^\d{10}$', elem.text):
                    result["supplier_vkn"] = elem.text
                    break

        if result["payment_amount"] == "Not found":
            for elem in root.iter():
                if elem.text and re.search(r'[\d.,]+ TRY', elem.text):
                    result["payment_amount"] = elem.text
                    break

        if result["id"] == "Not found":
            for elem in root.iter():
                # ID appears to be 16 digits
                if elem.text and re.match(r'^\d{16}$', elem.text):
                    result["id"] = elem.text
                    break

        # Print the full XML structure for debugging
        # print("\nXML structure:")
        for elem in root.iter():
            if elem.text and elem.text.strip():
                # print(f"{elem.tag} = {elem.text.strip()}")
                a = 1
        return str(result)

    except Exception as e:
        print(f"Error extracting data: {e}")
        return None


# Usage
if __name__ == "__main__":
    # Replace with your XML file path
    xml_file_path = "data/xml/49E6D63B-F491-1FD0-82F8-B838D7067E5C.XML"

    data = extract_data_from_xml(xml_file_path)
    # print("\nExtracted Data:")
    # print(f"Supplier VKN: {data['supplier_vkn']}")
    # print(f"Payment Amount: {data['payment_amount']}")
    # print(f"ID: {data['id']}")
    print(data)
