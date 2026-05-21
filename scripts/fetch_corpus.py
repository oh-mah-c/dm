import urllib.request, json, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

output = r'e:\Folder_Code\dm\datasets\tokenizer\real_corpus.txt'

articles = [
    'Machine_learning', 'Natural_language_processing', 'Transformer_(machine_learning_model)',
    'BERT_(language_model)', 'GPT-4', 'Tokenization_(lexical_analysis)',
    'Byte_pair_encoding', 'Neural_machine_translation', 'Large_language_model',
    'Information_theory', 'Entropy_(information_theory)', 'Optimal_transport',
    'Unsupervised_learning', 'Deep_learning', 'Recurrent_neural_network',
    'Attention_(machine_learning)', 'Word2vec', 'Text_segmentation',
    'Statistical_machine_translation', 'Language_model', 'Vocabulary',
    'Linguistics', 'Computational_linguistics', 'Phonology', 'Morphology_(linguistics)',
    'Syntax', 'Semantics', 'Pragmatics', 'Corpus_linguistics',
]

lines = []
for art in articles:
    url = (f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts'
           f'&exintro=0&format=json&titles={art}&redirects=1')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'dm-tokenizer-research/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode('utf-8'))
        pages = data['query']['pages']
        for pid, page in pages.items():
            if 'extract' in page:
                text = page['extract']
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                for sent in re.split(r'(?<=[.!?]) +', text):
                    sent = sent.strip()
                    if len(sent) > 20:
                        lines.append(sent)
        print(f'OK: {art} ({len(lines)} lines so far)')
    except Exception as e:
        print(f'SKIP {art}: {e}')

with open(output, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Written {len(lines)} lines to {output}')
