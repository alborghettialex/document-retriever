import fitz
from abc import ABC, abstractmethod

class TextExtractor(ABC):
    """
    Abstract base class for text extraction from documents.
    """
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        ...

class PDFTextExtractor(TextExtractor):
    """
    Text extractor for PDF documents.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extracts text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file.

        Returns: 
            str: Extracted text as a string.
        """
        document = fitz.open(file_path)
        text = ""
        for page in document:
            text += page.get_text()
        document.close()
        return text

class TXTTextExtractor(TextExtractor):
    """
    Text extractor for TXT documents.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extracts text from a TXT file.

        Args:
            file_path (str): Path to the TXT file.

        Returns: 
            str: Extracted text as a string.
        """
        with open(file_path) as fp:
            text = fp.read()
        return text