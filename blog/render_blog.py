"""Build main.html from main.md. Dependencies: markdown-it-py, beautifulsoup4.
Usage: python render_blog.py [source.md] [output.html]
The Markdown stays unchanged; selected references become HTML margin notes.
"""
from pathlib import Path
import re, html, sys
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('main.md')
output=Path(sys.argv[2]) if len(sys.argv)>2 else source.with_suffix('.html')
text=source.read_text()
math=[]
def protect(m):
    math.append(m.group())
    return f'MATHPLACEHOLDER{len(math)-1}END'
# Preserve math through Markdown parsing, including backslash delimiters.
text=re.sub(r'\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|(?<!\\)\$(?!\$)[^$\n]+?\$',protect,text)
body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
for i,formula in enumerate(math):
    display=formula.startswith('$$')
    raw=formula[2:-2] if display or formula.startswith('\\(') else formula[1:-1]
    body=body.replace(f'MATHPLACEHOLDER{i}END',f'<span class="math">{html.escape(("\\[" if display else "\\(")+raw+("\\]" if display else "\\)"))}</span>')
soup=BeautifulSoup(body,'html.parser')
# IDs preserve explicit Markdown anchors and remain stable across rebuilds.
used={e['id'] for e in soup.select('[id]')}
nav=[]
for h in soup.select('h2,h3'):
    label=h.get_text(' ',strip=True)
    slug=re.sub(r'[^a-z0-9]+','-',label.lower()).strip('-')
    while slug in used:slug+='-section'
    h['id']=slug;used.add(slug)
    if h.name=='h2':
        short_labels=['TL;DR', '1. Steering interpretation', '2. Pathological looping',
                      '3. Training the sensing vector', '4. Non-pathological setting',
                      '5. Data and sensing location', '6. Conclusion', '7. Notes']
        short=short_labels[len(nav)] if len(nav)<len(short_labels) else label
        h["data-nav-label"]=short  # lets other hosts (the personal site) reuse the short menu names
        nav.append((h.name,slug,short))
# Anchors for bold run-in notes.
for p in soup.find_all('p',recursive=False):
    strong=p.find('strong',recursive=False)
    if strong and p.contents[0]==strong:
        label=strong.get_text().rstrip('.?')
        slug=re.sub(r'[^a-z0-9]+','-',label.lower()).strip('-')
        if slug not in used:p['id']=slug;used.add(slug)
notes=[]
def note_after(block,content):
    n=len(notes)+1
    marker=soup.new_tag('a',href=f'#margin-note-{n}',id=f'note-ref-{n}')
    marker['class']='note-ref';marker['aria-label']=f'Read note {n}';marker.string=str(n)
    block.append(marker)
    aside=soup.new_tag('aside',id=f'margin-note-{n}')
    aside['class']='margin-note';aside['role']='note';aside['aria-label']=f'Note {n}'
    back=soup.new_tag('a',href=f'#note-ref-{n}');back['class']='note-number';back['aria-label']=f'Back to note {n} reference';back.string=str(n)
    aside.append(back)
    fragment=BeautifulSoup(content,'html.parser')
    for node in list(fragment.contents):aside.append(node)
    block.insert_after(aside);notes.append(n)
# Explicit Markdown side notes become margin notes without rewriting adjacent prose.
for block in list(soup.find_all('blockquote', recursive=False)):
    p = block.find('p')
    if not p or not p.get_text().startswith('Side note — '):
        continue
    label = p.find('strong')
    if label:
        label.string = label.get_text().replace('Side note — ', '', 1)
    anchor = block.find_previous_sibling('p')
    if anchor:
        note_after(anchor, p.decode_contents())
        block.decompose()
# Number notes in reading order, not transformation order.
pairs=[(a,soup.find('a',href='#'+a['id'])) for a in soup.select('.margin-note')]
for n,(aside,marker) in enumerate(pairs,1):
    marker['href']=f'#margin-note-{n}';marker['id']=f'note-ref-{n}';marker.string=str(n)
    marker['aria-label']=f'Read note {n}'
    aside['id']=f'margin-note-{n}';aside['aria-label']=f'Note {n}'
    back=aside.select_one('.note-number');back['href']=f'#note-ref-{n}';back.string=str(n);back['aria-label']=f'Back to note {n} reference'
# Figures and their existing Markdown captions.
for p in list(soup.find_all('p',recursive=False)):
    img=p.find('img')
    if not img:continue
    fig=soup.new_tag('figure')
    a=soup.new_tag('a',href=img['src']);a['class']='figure-link';a['aria-label']='Open full-size figure'
    a.append(img.extract());fig.append(a)
    nxt=p.find_next_sibling()
    if nxt and nxt.name=='p' and nxt.find('em') and len(nxt.contents)==1:
        cap=soup.new_tag('figcaption')
        for c in list(nxt.em.contents):cap.append(c.extract())
        fig.append(cap);nxt.decompose()
    p.replace_with(fig)
for table in soup.find_all('table'):
    wrapper=soup.new_tag('div');wrapper['class']='table-scroll';wrapper['tabindex']='0';wrapper['role']='region';wrapper['aria-label']='Scrollable data table'
    table.wrap(wrapper)
title=soup.h1.get_text()
subtitle=soup.find('p',class_='subtitle')
if subtitle:title+=': '+subtitle.get_text()
links=''.join(f'<a class="nav-{tag}" href="#{slug}">{html.escape(label)}</a>' for tag,slug,label in nav)
css=r"""
:root{--ink:#26354a;--muted:#667587;--green:#217e71;--line:#dde5ea;--soft:#f5f8fa}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:36px}
body{margin:0;background:#fff;color:var(--ink);font:18px/1.75 Georgia,"Times New Roman",serif}
.layout{max-width:1496px;margin:auto;padding:64px 32px 100px;display:grid;grid-template-columns:200px minmax(0,1fr) 200px;gap:64px;align-items:start;justify-content:center}
article{grid-column:2;min-width:0;position:relative}
h1,h2,h3,nav,.margin-note,figcaption,table{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
h1{font-size:clamp(30px,2.6vw,42px);line-height:1.22;letter-spacing:-.035em;margin:0 0 24px;font-weight:700}
.subtitle{font:400 clamp(19px,1.5vw,23px)/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--muted);margin:-12px 0 24px}
.byline{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.5;color:var(--muted);margin:-4px 0 36px 0}
.authors{display:flex;flex-wrap:wrap;gap:10px 32px;margin-bottom:8px}
.authors .name{display:inline-block;color:var(--ink);text-decoration:underline;text-decoration-color:var(--line);text-underline-offset:4px;transition:color .15s,text-decoration-color .15s}
.authors a.name:hover{color:var(--green);text-decoration-color:var(--green)}
.byline .affil{font-size:14px}
h2{font-size:27px;line-height:1.3;margin:64px 0 24px;letter-spacing:-.025em}
h3{font-size:21px;line-height:1.4;margin:40px 0 18px}
p{margin:0 0 24px}li{margin:8px 0}a{color:var(--green);text-decoration-thickness:1px;text-underline-offset:3px}
a:hover{color:#144e46}a:focus-visible,button:focus-visible{outline:3px solid #279c88;outline-offset:4px}
blockquote{margin:28px 0;padding:8px 24px;border-left:3px solid #a6cfc7;background:#f8faf9}
.example-positive{color:#16735f}.example-negative{color:#a83d49}
blockquote p:last-child{margin-bottom:8px}
code{font: .82em/1.5 ui-monospace,monospace;background:var(--soft);padding:2px 4px;overflow-wrap:anywhere}
nav{grid-column:1;position:sticky;top:32px;font-size:13px;line-height:1.45;max-height:calc(100vh - 64px);overflow-y:auto;padding:5px 12px 16px 0}
.nav-label{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 18px 10px}
nav a{display:block;text-decoration:none;color:var(--muted);padding:9px 10px;margin:3px 0;border-radius:6px;border-left:2px solid transparent;transition:transform .16s,box-shadow .16s,background .16s}
nav a:hover,nav a:focus-visible{transform:translateY(-2px);background:white;box-shadow:0 5px 16px #26354a14;color:var(--ink)}
nav a[aria-current="location"]{color:var(--green);background:#edf6f3;border-left-color:var(--green)}
nav .nav-h3{padding-left:20px;font-size:12px}
#nav-toggle{display:none}
figure{margin:34px 0}figure img{display:block;width:100%;height:auto}
.figure-link{display:block;cursor:zoom-in}
figcaption{font-size:13px;line-height:1.65;color:var(--muted);margin-top:14px}
.table-scroll{overflow-x:auto;margin:24px 0;max-width:100%;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;font-size:13px;line-height:1.55;width:100%}
th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;min-width:100px}
th{background:var(--soft);font-weight:600}tr:last-child td{border-bottom:0}
.math{overflow-wrap:normal}mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;padding:8px 0;max-width:100%}
.note-ref{font-family:ui-sans-serif,system-ui,sans-serif;vertical-align:super;font-size:11px;text-decoration:none;margin-left:5px;background:#edf6f3;padding:2px 5px;border-radius:3px}
.margin-note{font-size:13px;line-height:1.65;color:var(--muted);padding:12px 0 12px 14px;border-left:2px solid #c4d9d4;margin:16px 0 24px;background:white}
.margin-note strong{font-weight:600;color:var(--ink)}.note-number{font-weight:600;margin-right:7px;text-decoration:none}
.margin-note:target{background:#edf6f3}.margin-note p{margin:0}
.wide-notes .margin-note{position:absolute;left:calc(100% + 64px);width:200px;margin:0}
.skip{position:fixed;top:-100px;left:16px;padding:10px;background:white;z-index:10}.skip:focus{top:10px}
@media(min-width:1250px){.figure-link{width:110%;margin-left:-5%}figcaption{width:100%}}
@media(max-width:1249px){.layout{grid-template-columns:180px minmax(0,760px);max-width:1060px;gap:28px;padding:40px 24px}article{grid-column:2}.margin-note{background:var(--soft);padding:14px 18px}}
@media(max-width:850px){.layout{display:block;padding:0 20px 64px}nav{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(12px);max-height:55vh;padding:10px 0;border-bottom:1px solid var(--line);margin:0 -20px 32px;padding-left:20px;padding-right:20px}#nav-toggle{display:block;border:0;background:transparent;color:var(--ink);font:600 14px/1.6 ui-sans-serif,system-ui;cursor:pointer;width:100%;text-align:left}.nav-label{display:none}.js #nav-links{display:none}.js nav.open #nav-links{display:block}article{width:100%}body{font-size:17px}h1{font-size:31px}h2{font-size:25px;scroll-margin-top:70px}h3{scroll-margin-top:70px}.margin-note{font-size:13px}figure{margin-left:0;margin-right:0}th,td{min-width:0;padding:10px 8px}.figure-link::after{content:"Tap to enlarge";display:block;font:12px/1.4 ui-sans-serif,system-ui,sans-serif;color:var(--muted);text-align:right;margin-top:6px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}nav a{transition:none;transform:none!important}}
@media print{nav,.skip{display:none}.layout{display:block;padding:0}article{width:100%}.wide-notes .margin-note{position:static;width:auto;margin:12px 0}a{color:inherit}h2,h3{break-after:avoid}figure{break-inside:avoid}.figure-link{width:100%;margin-left:0}}
"""
js=r"""
document.documentElement.classList.add('js');
const article=document.querySelector('article'), notes=[...document.querySelectorAll('.margin-note')];
let pending=false;
function placeNotes(){
 pending=false;
 const wide=window.innerWidth>=1250;
 article.classList.toggle('wide-notes',wide);
 let bottom=0;
 for(const note of notes){
  if(!wide){note.style.top='';continue}
  const ref=document.getElementById(note.id.replace('margin-note','note-ref'));
  const wanted=ref.getBoundingClientRect().top-article.getBoundingClientRect().top-8;
  const top=Math.max(wanted,bottom);
  note.style.top=top+'px';bottom=top+note.offsetHeight+24;
 }
}
function scheduleNotes(){if(!pending){pending=true;requestAnimationFrame(placeNotes)}}
window.addEventListener('resize',scheduleNotes);
window.addEventListener('load',scheduleNotes);
document.fonts.ready.then(scheduleNotes);
document.querySelectorAll('img').forEach(img=>img.addEventListener('load',scheduleNotes));
const resizeObserver=new ResizeObserver(scheduleNotes);resizeObserver.observe(article);notes.forEach(n=>resizeObserver.observe(n));
const heads=[...document.querySelectorAll('article h2')], navLinks=[...document.querySelectorAll('#nav-links a')];
function activeSection(){
 let current=heads[0];for(const h of heads){if(h.getBoundingClientRect().top<=150)current=h;else break}
 navLinks.forEach(a=>{if(current&&a.hash==='#'+current.id)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});
}
window.addEventListener('scroll',activeSection,{passive:true});activeSection();placeNotes();
const toggle=document.getElementById('nav-toggle'),nav=document.querySelector('nav');
toggle.addEventListener('click',()=>{const open=nav.classList.toggle('open');toggle.setAttribute('aria-expanded',String(open))});
navLinks.forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');toggle.setAttribute('aria-expanded','false')}));
// Shrink display equations that are slightly too wide (down to 75%); wider ones still scroll sideways.
function fitEquations(){
 document.querySelectorAll('mjx-container[display="true"]').forEach(m=>{
  m.style.fontSize='';
  if(m.scrollWidth<=m.clientWidth)return;
  const scale=Math.max(.75,m.clientWidth/m.scrollWidth*.98);
  m.style.fontSize=parseFloat(getComputedStyle(m).fontSize)*scale+'px';
 });
}
window.addEventListener('resize',fitEquations);
window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},startup:{ready(){MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>{fitEquations();scheduleNotes()})}}};
"""
page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{css}</style></head>
<body><a class="skip" href="#article">Skip to article</a><div class="layout">
<nav aria-label="Article sections"><button id="nav-toggle" aria-expanded="false" aria-controls="nav-links">Sections ▾</button><div id="nav-links">{links}</div></nav>
<article id="article">{soup}</article></div><script>{js}</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
</body></html>'''
output.write_text(page)
print(f'Built {output.name}: {len(nav)} navigation entries, {len(notes)} margin notes.')
