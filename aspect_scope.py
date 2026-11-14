"""Aspect-local context features. The dictionaries come from assignment definitions.

They select input clauses; final labels are learned from train.csv, never rules.
"""
import re,unicodedata
from absa_core import normalize

LEXICONS={
'nhan_vien':r'nhan vien|le tan|phuc vu|thai do|nhiet tinh|than thien|tiep tan|bao ve|check.?in|check.?out|nhan phong|tra phong|xu ly|hoa don|dich vu',
'sach_se':r'sach|ban|do ban|ve sinh|mui|am moc|con trung|muoi|kien|gian|ruoi|bui|thom|ga goi',
'vi_tri':r'vi tri|dia diem|trung tam|bien|di chuyen|duong|xung quanh|khu vuc|gan|xa|tien di|ben ngoai',
'phong':r'phong|giuong|nem|dem|noi that|dien tich|dieu hoa|may lanh|tivi|\btv\b|nuoc nong|voi sen|bon tam|ban cong|view|cach am|ngu|nha tam|phong tam|tam nhin',
'gia_tri':r'gia|tien|phi|dat|re|hop ly|hop li|dang|phu thu|hoa don|thanh toan|chi phi',
'an_uong':r'an|uong|bua sang|buffet|buffe|nha hang|thuc don|do an|do uong|mon|ca phe|cafe|breakfast|\bbf\b|\bbar\b',
'tien_nghi':r'ho boi|be boi|boi|spa|gym|wifi|wi.?fi|internet|tien nghi|tien ich|thang may|do xe|dau xe|bai xe|san|khuon vien|tour|dua don|xe dap|khu vui choi|sanh|hoat dong'}

def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD',text.lower()) if unicodedata.category(c)!='Mn').replace('đ','d')

PATTERNS={a:re.compile(r'(?<!\w)(?:'+s+r')(?!\w)') for a,s in LEXICONS.items()}

def scoped_texts(texts):
    result={a:[] for a in LEXICONS}
    for raw in texts:
        text=normalize(raw,'lexical')
        clauses=[c.strip() for c in re.split(r'[.!?;\n,]+|\bnhưng\b|\btuy nhiên\b',text) if c.strip()]
        hits=[{a for a,p in PATTERNS.items() if p.search(fold(c))} for c in clauses]
        for a in LEXICONS:
            selected=[]
            for i,c in enumerate(clauses):
                if a in hits[i] or (not hits[i] and i>0 and a in hits[i-1]):selected.append(c)
            result[a].append(' . '.join(selected) or 'khong_co_ngu_canh')
    return result
