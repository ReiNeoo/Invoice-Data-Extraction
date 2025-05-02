from abc import ABC, abstractmethod


class DataExtractor(ABC):
    """Abstract base class defining the interface for all extractors"""

    @abstractmethod
    def extract(self, source):
        """Extract data from the source and return structured information"""
        pass

    @abstractmethod
    def get_information(self, raw_text):
        """Process extracted text into structured data"""
        pass
