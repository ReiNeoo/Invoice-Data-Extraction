from bs4 import BeautifulSoup
import requests
from src.extract_data_image import ExtractDataFromImageYOLO
import re
import time


def verify(captcha):
    url = "https://ebelge.gib.gov.tr/earsivsorgula.php"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://ebelge.gib.gov.tr",
        "Referer": "https://ebelge.gib.gov.tr/earsivsorgula.php",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }

    cookies = {
        # Tarayıcıdan kopyaladığın aktif session çerezi
        "PHPSESSID": "m8p4iq2vt8dknffeq3rf5p88oj",
        "_ga": "GA1.3.1744025693.1...",
        "_ga_YPR8T63RR": "GS1.1.1744025697.0.0.0"
    }

    data = {
        "imageId": "2",
        "SATICIVKN": "0080153737",
        "FATURAIID": "AAA2025002599812",
        "ODTUTAR": "1.484,10",
        "ODTUTARDisplayed": "1.484,10",
        "captcha_code": captcha,  # Bu değişecek, dinamik çözüm gerekir
        "submit": ""
    }

    response = requests.post(url, headers=headers, cookies=cookies, data=data)
    soup = BeautifulSoup(response.text, "html.parser")

    # Get the div containing the result
    result_div = soup.find("div", {"id": "showSorgusonuc"})
    print(result_div)

    if result_div:
        text = result_div.get_text(separator=" ", strip=True)
        print(text)

        # Optional: Regex to extract structured data
        fatura_no = re.search(r"Sorgulanan\s+(\w+)", text)
        vkn = re.search(r"Vergi Kimlik Numarası\s+(\d+)", text)
        tutar = re.search(r"Ödenecek Tutarı\s+([\d.,]+)", text)
        alici = re.search(r"Alıcısı\s+(.+?)\s+Olan Fatura", text)
        status = re.search(r"Fatura\s+(.+)$", text)

        print("Fatura No:", fatura_no.group(1) if fatura_no else None)
        print("Satıcı VKN:", vkn.group(1) if vkn else None)
        print("Tutar:", tutar.group(1) if tutar else None)
        print("Alıcı:", alici.group(1) if alici else None)
        print("Durum:", status.group(1) if status else None)
    else:
        print("No result found.")


def get_captcha():

    text_extractor = ExtractDataFromImageYOLO()

    timestamp = int(time.time() * 1000)

    url = f"https://ebelge.gib.gov.tr/earsivsorgula.php?cfimg={timestamp}"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Referer": "https://ebelge.gib.gov.tr/earsivsorgula.php"
    }

    cookies = {
        "PHPSESSID": "m8p4iq2vt8dknffeq3rf5p88oj",  # Aynı session cookie'si
        # Diğer çerezleri de eklemen gerekebilir, isteğe göre güncelle.
    }

    response = requests.get(url, headers=headers, cookies=cookies)

    if response.status_code == 200:
        with open("captcha.jpg", "wb") as f:
            f.write(response.content)
        captcha = text_extractor.extract_text_with_OCR_captcha(
            response.content)
        print("CAPTCHA indirildi.")
        print(f"CAPTCHA: {captcha}")
    else:
        print("Hata:", response.status_code)

    return captcha


cap = get_captcha()
ver = verify(cap)
