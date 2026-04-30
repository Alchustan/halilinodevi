import json
import re
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor

def translate_text(text):
    if not text.strip() or text.strip() in ['---', '']:
        return text
    try:
        translator = GoogleTranslator(source='en', target='tr')
        # Split text into chunks if it's too long, but usually markdown cells are fine
        return translator.translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

notebook_path = '/Users/baris/Projects/halilinodevi/odev.ipynb'
output_path = '/Users/baris/Projects/halilinodevi/odev_tr.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

def process_cell(cell):
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            # Translate markdown lines
            if line.strip():
                translated_line = translate_text(line)
                new_source.append(translated_line + ('\n' if line.endswith('\n') else ''))
            else:
                new_source.append(line)
        cell['source'] = new_source
    elif cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Translate python comments
            if line.strip().startswith('#'):
                comment_text = line.split('#', 1)[1]
                translated_comment = translate_text(comment_text)
                indent = line[:line.find('#')]
                new_source.append(indent + '#' + translated_comment + ('\n' if line.endswith('\n') else ''))
            else:
                new_source.append(line)
        cell['source'] = new_source

print("Starting translation...")
# We use sequential processing to not hit rate limits too hard, but let's just do sequential
for i, cell in enumerate(nb['cells']):
    print(f"Translating cell {i+1}/{len(nb['cells'])}...")
    process_cell(cell)

with open(output_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully translated notebook and saved to {output_path}")
