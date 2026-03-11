from io import BytesIO

import requests
from pypdf import PdfReader
from selenium.webdriver.common.by import By


class ContentHandling:
    @staticmethod
    def handle_pdf(url: str) -> str:
        """Extract text from PDF documents."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with BytesIO(response.content) as pdf_file:
                reader = PdfReader(pdf_file)
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text[:2000]
        except Exception as exc:
            return f"Failed to extract PDF: {exc}"

    @staticmethod
    def handle_arxiv(driver) -> str:
        """Extract abstract from arXiv pages."""
        try:
            abstract = driver.find_element(By.CSS_SELECTOR, "blockquote.abstract").text
            return abstract.replace("Abstract: ", "")
        except Exception:
            return driver.find_element(By.TAG_NAME, "body").text[:1000]

    @staticmethod
    def postprocess_if_special(result: dict, driver) -> dict:
        url = result.get("link", "")
        if url.endswith(".pdf"):
            result["text"] = ContentHandling.handle_pdf(url)
        elif "arxiv.org" in url:
            result["text"] = ContentHandling.handle_arxiv(driver)
        return result
