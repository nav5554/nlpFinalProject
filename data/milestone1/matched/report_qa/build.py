import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

root=Path("data/milestone1")
folder=root/"matched"
r=json.loads((folder/"results.json").read_text())
d=Document()
d.core_properties.title="When to Remove the Mask"
d.core_properties.author="Arnav Sehgal and Tobias Thurston"
sec=d.sections[0]
sec.top_margin=Inches(.70)
sec.bottom_margin=Inches(.65)
sec.left_margin=sec.right_margin=Inches(.86)
for st,size in [('Normal',11),('Title',29),('Heading 1',20),('Heading 2',14)]:
    style=d.styles[st]; style.font.name='Calibri'; style.font.size=Pt(size)
    style.font.color.rgb=RGBColor.from_string('171D22' if st=='Normal' else '000000')
    if st!='Normal': style.font.bold=True
    pf=style.paragraph_format
    pf.space_after=Pt(5 if st=='Normal' else 4)
    pf.space_before=Pt(0 if st in ('Title','Normal') else 8)
    pf.line_spacing=1.08
def p(s): d.add_paragraph(s)
def h(s): d.add_heading(s,2)
def note(s):
    pp=d.add_paragraph(); rr=pp.add_run(s); rr.italic=True; rr.font.size=Pt(9.5)
    rr.font.color.rgb=RGBColor.from_string('374955')
def table(headers,rows,widths):
    t=d.add_table(rows=0,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for c,w in zip(t.columns,widths): c.width=Inches(w)
    for i,vals in enumerate([headers]+rows):
        cells=t.add_row().cells
        for j,(c,val,w) in enumerate(zip(cells,vals,widths)):
            c.width=Inches(w); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pp=c.paragraphs[0]; pf=pp.paragraph_format; pf.space_before=Pt(3); pf.space_after=Pt(3); pf.line_spacing=1.0
            if j>0: pp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
            rr=pp.add_run(str(val)); rr.bold=i==0; rr.font.size=Pt(10)
            if i==0 or i%2==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'D6E2E5' if i==0 else 'F0F4F4');c._tc.get_or_add_tcPr().append(shade)
title_p=d.styles["Title"].element.find(qn("w:pPr"))
if title_p is not None:
    for border in title_p.findall(qn("w:pBdr")): title_p.remove(border)
heading=d.add_paragraph(style="Title");heading.add_run("When to Remove the Mask")
note("COMS 4705  ·  Final Project Milestone 1  ·  Arnav Sehgal and Tobias Thurston")
h("Research question and proposal feedback")
p("Can a small neural ranker recover more missing tokens than confidence-first decoding by changing which guesses a frozen masked diffusion language model commits first? Following the TA’s feedback, we masked complete middle sentences with visible context on both sides. On fresh articles the sentence-trained wider MLP recovered 104 tokens versus 78 for confidence-first. Logistic regression recovered 108 and one-shot recovered 173.")
h("What the learned policy does")
p("We use the 169.6M-parameter OpenWebText MDLM checkpoint from Sahoo et al. (2024) with a GPT-2 tokenizer. At each pass MDLM proposes one token per remaining blank. The policy assigns each proposal a score. The decoder commits the highest-scoring positions and keeps those tokens fixed for subsequent passes.")
p("The ranker sees 17 standardized features. Three describe prediction confidence (top probability, top-two margin, entropy). Four track stability (argmax flip, stable-step count, confidence change, first-step flag). Six describe the masked span (distance to visible text, run length, left/right mask flags and neighbor confidences). Four give within-pass context (confidence rank, deviation from mean confidence, fraction masked, fraction of passes elapsed).")
p("Each training row labels whether the current argmax matches the reference token. We collect rows along confidence-first and random-order trajectories. Logistic regression and a two-hidden-layer ReLU MLP score the same features. During decoding, each method commits the ceiling of remaining blanks divided by remaining passes. All iterative methods share the same starting mask and 32-pass budget.")
h("Results across masking settings")
rows=[
["Multiple spans · test","300","19.9","17.4","17.4","17.8","18.2"],
["64-token gap · validation","24","8.27","4.10","4.04","4.30","4.82"],
["Whole sentence · validation","48","7.90","4.80","5.16","5.52","4.49"]]
table(["Setting","Windows","1 pass","Conf.","Edges","LR","MLP"],rows,[2.32,.82,.71,.65,.63,.60,.65])
note("Masked-token exact match (%). The first row uses scorers trained on multiple spans. The next two use the same fixed-gap scorers. The scored token counts are 33,600, 1,536 and 1,937, respectively.")
p("The MLP gained 11 exact matches over confidence-first on the 64-token gap. On whole sentences it lost six. The paired difference was −0.31 percentage points, with a 95% article-bootstrap interval of −1.30 to +0.54. This motivated a direct test of sentence-specific training.")
p("An earlier 50% random-mask control used 40 test windows and 5,151 blanks. One-shot scored 40.8%, confidence-first 40.4%, and the greedy answer-key policy 51.4%. The oracle always fills its quota, including wrong guesses when needed. Its score diagnoses sensitivity to commitment order.")
note("Sahoo et al., Simple and Effective Masked Diffusion Language Models (NeurIPS 2024). WikiText-2 raw is from Merity et al. Code and results use nav5554/nlpFinalProject at 1475d64 and the local experiment in data/milestone1/matched/.")
d.add_page_break()
d.add_heading("Whole sentence training and model size",1)
p(f"We trained on 64 middle sentences from 64 WikiText-2 training articles and used 16 validation sentences for checkpoint selection. Before training, we reserved {r['passages']} nonoverlapping validation passages from {r['articles']} articles outside the earlier local pilots. Each 256-token passage contains one missing middle sentence of 32–61 GPT-2 tokens. Every ranker saw the same {r['masked_tokens']:,} blanks. Seed 53 fixed passage selection and seed 0 controlled training.")
labels=[("one_shot","One-shot MDLM"),("confidence","Confidence-first"),("edges","Edges-first"),("prior_lr","Logistic regression · gap training"),("prior_mlp","MLP 64 · gap training"),("lr","Logistic regression · sentence training"),("mlp","MLP 64 · sentence training"),("wide","MLP 128 · sentence training"),("oracle","Greedy answer-key diagnostic")]
table(["Method","Passes","Correct tokens","Accuracy"],[[name,1 if key=="one_shot" else 32,r['hits'][key],f"{r['accuracy'][key]*100:.2f}%"] for key,name in labels],[3.49,.78,1.15,.96])
h("What the comparisons resolve")
cmp=r['comparisons_to_mlp64']
def interval(m):
    lo,hi=cmp[m]['article_bootstrap_95_pp'];return f"{lo:+.2f} to {hi:+.2f}"
p("With sentence training, the 128-wide MLP beat confidence-first by 1.31 percentage points (95% paired article-bootstrap interval +0.61 to +2.09). Logistic regression recovered four more tokens than the wider MLP. Their difference was −0.20 points for the MLP (−1.11 to +0.66). One-shot recovered the most reference tokens among methods without access to the answer key.")
audit=json.loads((folder/"audit.json").read_text())
paired=audit["paired_differences_pp"]["wide minus mlp"]
wide_lo,wide_hi=paired["ci95"]
note(f"Doubling both MLP widths increased the parameter count from 5,377 to 18,945 and tuning within-step AUROC from 0.666 to 0.682. The decoding gain was +0.65 points, with a 95% interval of {wide_lo:+.2f} to {wide_hi:+.2f} points. AUROC measures guess ranking within a pass, while exact match scores the completed gap.")
k=r['initial_topk_hits']
p(f"Before any commits, the frozen MDLM placed the reference token among its top ten for {k['10']}/{r['masked_tokens']} blanks ({100*k['10']/r['masked_tokens']:.1f}%). The policy can commit only the current top guess. This diagnostic measures candidate availability at the initial step. Testing a second frozen generator would be necessary to attribute any difference to backbone capacity.")
p("In validation row 3701, the reference contains the phrase “reported that number”. At its final position, the wider MLP committed “number” and confidence-first committed “for”. Across this sentence they recovered three tokens and one token, respectively.")
h("Limits and remaining work")
p("Exact match measures recovery of the original wording. Several different sentences can fit the same context. The sentence evaluation covers long sentences, one training seed and a small set of articles. The gap-trained and sentence-trained scorers also saw different training windows and row counts.")
p("For the final report we plan to evaluate a ranking loss that compares blanks within the same window and step, where the commit decision occurs. We will use validation for model selection, repeat training across seeds and report paired article-level uncertainty. A blinded review of passage coherence will complement token recovery.")
d.save(folder/'NLP Milestone 1.docx')
