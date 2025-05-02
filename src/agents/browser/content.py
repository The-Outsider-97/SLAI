import requests
from pypdf import PdfReader
from io import BytesIO
from selenium.webdriver.common.by import By 

class ContentHandling:
    def handle_pdf(url):
        """Extract text from PDF documents"""
        try:
            response = requests.get(url, timeout=10)
            with BytesIO(response.content) as pdf_file:
                reader = PdfReader(pdf_file)
                text = "\n".join([page.extract_text() for page in reader.pages])
            return text[:2000]  # Return first 2000 characters
        except Exception as e:
            return f"Failed to extract PDF: {str(e)}"

    def handle_arxiv(driver):
        """Extract abstract from arXiv pages"""
        try:
            abstract = driver.find_element(By.CSS_SELECTOR, "blockquote.abstract").text
            return abstract.replace("Abstract: ", "")
        except:
            return driver.find_element(By.TAG_NAME, "body").text[:1000]
        
    def _postprocess_if_special(self, result, driver):
        url = result.get("link", "")
        if url.endswith(".pdf"):
            result["text"] = self.handle_pdf(url)
        elif "arxiv.org" in url:
            result["text"] = self.handle_arxiv(driver)
        return result
