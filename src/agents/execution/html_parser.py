

from html.parser import HTMLParser

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("HTML Parser")
printer = PrettyPrinter

class HTMLParser(HTMLParser):
    """Extended HTML parser with support for common semantic elements"""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.parser_config = get_config_section('html_parser')
        self.structure = {
            'title': '',
            'links': [],
            'metadata': {},
            'headings': {f'h{i}': [] for i in range(1, 7)},
            'scripts': [],
            'images': []
        }
        self.current_tag = None
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs = dict(attrs)
        
        if tag == 'title':
            pass
        elif tag == 'a' and 'href' in attrs:
            self.structure['links'].append(attrs['href'])
        elif tag in self.structure['headings']:
            pass
        elif tag == 'meta' and ('name' in attrs or 'property' in attrs):
            key = attrs.get('name') or attrs.get('property')
            self.structure['metadata'][key] = attrs.get('content', '')
        elif tag == 'script' and 'src' in attrs:
            self.structure['scripts'].append(attrs['src'])
        elif tag == 'img' and 'src' in attrs:
            self.structure['images'].append(attrs['src'])
    
    def handle_data(self, data):
        if self.current_tag == 'title':
            self.structure['title'] += data
        elif self.current_tag in self.structure['headings']:
            self.structure['headings'][self.current_tag].append(data.strip())
    
    def handle_endtag(self, tag):
        self.current_tag = None

if __name__ == "__main__":
    print("\n=== Running HTML Parser Test ===\n")
    printer.status("Init", "HTML Parser initialized", "success")

    parser = HTMLParser()
    print(parser)

    print("\n=== Simulation Complete ===")
