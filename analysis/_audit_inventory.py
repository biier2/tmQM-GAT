# -*- coding: utf-8 -*-
"""Main-text audit - inventory build: pull out every sentence containing numbers, every cell of every table,
and every Section/Table/Figure pointer in the main text, and write them to files for item-by-item recomputation."""
import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')
import docx

HERE = r'D:/ml/paper/draft/round6'
OUT = r'D:/ml/chem_ml5/_audit'
os.makedirs(OUT, exist_ok=True)
d = docx.Document(os.path.join(HERE, 'final_ACS_r6.docx'))
P = [p.text for p in d.paragraphs]

NUM = re.compile(r'\d')
# ---------- 1. sentences containing numbers/pointers ----------
sent_re = re.compile(r'(?<=[.;:!?])\s+(?=[A-Z(\u0394])')
with open(os.path.join(OUT, 'sent.txt'), 'w', encoding='utf-8') as f:
    n = 0
    for i, t in enumerate(P):
        if not t.strip():
            continue
        # split into sentences (keeping the delimiter)
        parts = re.split(r'(?<=[.;])\s+', t)
        for s in parts:
            if NUM.search(s) or re.search(r'Section \d|Table S?\d|Figure \d', s):
                f.write('P%03d| %s\n' % (i, s.strip()))
                n += 1
    print('%d sentences with numbers/pointers -> _audit/sent.txt' % n)

# ---------- 2. tables ----------
with open(os.path.join(OUT, 'tables.txt'), 'w', encoding='utf-8') as f:
    for ti, tb in enumerate(d.tables):
        f.write('\n' + '=' * 90 + '\nTable %d  (%d x %d)\n' % (ti + 1, len(tb.rows), len(tb.columns)))
        for r in tb.rows:
            f.write(' | '.join(c.text.replace('\n', ' ') for c in r.cells) + '\n')
    print('%d tables -> _audit/tables.txt' % len(d.tables))

# ---------- 3. pointers ----------
ptr = re.compile(r'(Section[s]? [^),;.]{1,30}?|Table S?\d[^),;.]{0,20}|Figure \d[^),;.]{0,20}|'
                 r'Supporting Information[^),;.]{0,40})')
with open(os.path.join(OUT, 'ptr.txt'), 'w', encoding='utf-8') as f:
    for i, t in enumerate(P):
        for m in ptr.finditer(t):
            a, b = max(0, m.start() - 55), min(len(t), m.end() + 40)
            f.write('P%03d| %-34s … %s …\n' % (i, m.group(1).strip(),
                                               t[a:b].replace('\n', ' ')))
    print('pointer list -> _audit/ptr.txt')

# ---------- 4. paragraph length table (find anomalies / short paragraphs) ----------
with open(os.path.join(OUT, 'paras.txt'), 'w', encoding='utf-8') as f:
    for i, t in enumerate(P):
        f.write('P%03d %5d  %s\n' % (i, len(t), t[:90].replace('\n', ' ')))
print('paragraph table -> _audit/paras.txt')
