"""Build the A1 novelty poster, two-page report and an offline recovery/study guide.
Uses the official header logos; other layout styles are allowed by the supplied instructions.
Complete report/novelty/poster_details.json and rerun to obtain a print-ready filename.
"""
from pathlib import Path
import json, html, shutil
import pandas as pd
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A1,A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF
import pymupdf
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'report/novelty'
DETAILS=json.loads((D/'poster_details.json').read_text())
TEAL=colors.HexColor('#167e76');INK=colors.HexColor('#163141');PALE=colors.HexColor('#edf5f5')

def summary(h):
 p=ROOT/('results/novelty_finance' if h==1 else 'results/novelty_finance_mean5')
 df=pd.read_csv(p/'metrics.csv');q=df[df.domain=='target'].groupby('method')[['RMSE','MAE','MSE']].mean()
 return p,q


def gain(q,a,b,col='RMSE'):return 100*(1-q.loc[a,col]/q.loc[b,col])


def p(text,size=10,color=INK,leading=None):
 return Paragraph(text,ParagraphStyle('p',fontName='Helvetica',fontSize=size,leading=leading or size*1.35,textColor=color,spaceAfter=5))


def table(rows,widths,size=9):
 t=Table([[p(str(x),size,colors.white if i==0 else INK) for x in row] for i,row in enumerate(rows)],colWidths=widths)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('ROWBACKGROUNDS',(0,1),(-1,-1),[PALE,colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 return t


def rows(q):
 names=['Persistence','Mean 60','Mean 20','HAR','TCN source','TCN both','Hybrid source','Hybrid both']
 return [['Model','RMSE','MAE']]+[[n,f'{q.loc[n,"RMSE"]:.4f}',f'{q.loc[n,"MAE"]:.4f}'] for n in names]


def drawp(c,text,x,y,w,size=21):
 obj=p(text,size);_,hh=obj.wrap(w,3000);obj.drawOn(c,x,y-hh);return y-hh-12


def heading(c,text,x,y,w):
 c.setFillColor(TEAL);c.rect(x,y-4,w,3,fill=1,stroke=0)
 return drawp(c,text,x,y-18,w,27)


def qrcode(c,url,x,y,size):
 code=qr.QrCodeWidget(url);bounds=code.getBounds();w=bounds[2]-bounds[0];h=bounds[3]-bounds[1]
 d=Drawing(size,size,transform=[size/w,0,0,size/h,0,0]);d.add(code);renderPDF.draw(d,c,x,y)


def main():
 paths={};qs={}
 for h in [1,5]:paths[h],qs[h]=summary(h)
 q=qs[5];q1=qs[1]
 legacy=pd.read_csv(ROOT/'results/novelty_finance/legacy_matched_metrics.csv').groupby('method')[['RMSE','MAE']].mean()
 legacy5=pd.read_csv(paths[5]/'legacy_matched_metrics.csv').groupby('method')[['RMSE','MAE']].mean()
 rtime=json.loads((paths[1]/'runtime.json').read_text())
 # Two-page report: deliberate page break makes the length auditable.
 body=[]
 body += [p('DeepGen | IE643 Novelty Assessment',19),p('Trading-range forecasting under sector shift',15,TEAL),p('Ayush Patel (23B0037) and Rishik Yadav (23B2103) | 18 November 2026',9),Spacer(1,8)]
 body += [p('<b>1. Motivation and scope.</b> The intensive project forecasts multivariate weather, transformer and financial series across domains. Financial gains were small: the earlier five-session model reduced global MSE by about 1.60% versus its input-mean baseline. We investigate an explicitly easier financial target: the next-session high-low range, followed by the average range over the next five sessions. The latter smooths unpredictable daily spikes and measures expected trading variability; neither task predicts profit or stock direction.',10)]
 body += [table([['Benchmark','Source to target','Data / split'],['Weather','Shimla to Delhi','2021-2024; hourly; 70/10/20'],['Energy','ETTh1 to ETTh2','2016-2018; hourly; 60/20/20'],['Finance','5 IT to 5 pharma stocks','2012-01-03 to 2025-06-27; 70/10/20']], [90,145,280]),Spacer(1,9)]
 body += [p('<b>2. Implemented novelty experiments.</b> The financial input is 60 trading sessions with return, high-low log range, overnight gap and log volume. The output is the mean of r = 100 ln(High/Low) over H = 1 or H = 5 future sessions. Each company is split chronologically. No window crosses company boundaries, and all forecast answers lie inside their assigned partition.',10)]
 body += [p('<b>Architecture.</b> A 2,449-parameter causal TCN (versus 26,888 neural parameters in the earlier full hybrid, with a narrower output task) uses width 16, kernel size 3 and dilations 1/2/4 with residual connections. Its output corrects the last five-session range average. A 14-feature ridge branch combines range averages over 1/5/20/60 sessions, range variability, and absolute return/gap statistics. This branch is HAR-inspired rather than a reproduction of the realized-volatility HAR model. A convex blend weight in {0, .25, .5, .75, 1} is selected on IT validation only.',10)]
 body += [p('<b>Scaling and objective.</b> Range, returns and gaps are divided by the preceding 60-session mean range; log volume is centered and scaled inside that input. Predictions are restored to range units. Source fitting minimizes range MSE. Adaptation minimizes source replay MSE + 0.1 MMD + 0.1 CORAL + 0.01 source-weight anchor. MMD uses five RBF bandwidths; CORAL aligns feature covariance. The anchor sums layer-wise mean squared weight differences.',10)]
 body += [p('<b>Strict label-free adaptation.</b> Only Pharma training inputs enter alignment. No Pharma forecasting answers or hidden-tail regression targets enter fitting or selection. Source validation chooses checkpoints; an alignment update is retained only if it improves source-validation MSE. The blend is kept identical across adaptation ablations. Controls include source-only, extra source replay, MMD only, CORAL only, both, ridge only and simple baselines.',10)]
 body += [p('<b>Reproducibility.</b> Three seeds (0/1/2), AdamW, batch 256, source learning rate .001, adaptation .0002, gradient clipping 1. H1: 24 source and 10 adaptation epochs. H5 mean: 16 and 6 epochs. Logs, protocols, input hashes, checkpoints and predictions are saved. Run python -m src.novelty_finance --horizon 1 or --horizon 5. Source validation tunes ridge strength. CPU execution uses four threads.',9),PageBreak()]
 body += [p('Results, interpretation and limitations',17),p('Errors are in percentage points of log range. Values average the three seed-wise test metrics. H1 and H5-mean are different targets; compare methods within each column.',9)]
 result=[['Model','H1 RMSE','H1 MAE','H5 mean RMSE','H5 mean MAE']]
 for n in ['Persistence','Mean 60','HAR','Hybrid source','Hybrid replay','Hybrid coral','Hybrid mmd','Hybrid both']:
  result.append([n,*[f'{qs[h].loc[n,col]:.4f}' for h in [1,5] for col in ['RMSE','MAE']]])
 result.append(['Earlier adapted (matched)', *[f'{v:.4f}' for v in [*legacy.loc['Earlier hybrid adapted_model'],*legacy5.loc['Earlier hybrid adapted_model']]]])
 body += [table(result,[135,85,85,105,105],8.5),Spacer(1,8)]
 body += [p(f'<b>Matched comparisons.</b> For H1, the new hybrid RMSE is {q1.loc["Hybrid both","RMSE"]:.4f}, compared with {legacy.loc["Earlier hybrid adapted_model","RMSE"]:.4f} for the first forecast step of the earlier adapted hybrid on identical origins. The respective MAEs are {q1.loc["Hybrid both","MAE"]:.4f} and {legacy.loc["Earlier hybrid adapted_model","MAE"]:.4f}. For the H5 mean target, RMSE improves by {gain(q,"Hybrid both","Persistence"):.2f}% versus persistence and {gain(q,"Hybrid both","Mean 60"):.2f}% versus the 60-session mean. The earlier model averaged over the same five sessions achieves RMSE {legacy5.loc['Earlier hybrid adapted_model','RMSE']:.4f} and MAE {legacy5.loc['Earlier hybrid adapted_model','MAE']:.4f}: it retains better RMSE, while the new specialist has lower MAE.',9.5)]
 body += [Image(str(paths[5]/'comparison.png'),width=450,height=207),Spacer(1,5)]
 same=all(abs(qs[h].loc['Hybrid both','RMSE']-qs[h].loc['Hybrid source','RMSE'])<1e-10 for h in [1,5])
 conclusion='All combined-alignment checkpoints reverted to the source checkpoint under the source-validation rule; no adaptation gain was demonstrated.' if same else f'Combined alignment changed H5 hybrid RMSE by {gain(q,"Hybrid both","Hybrid source"):.2f}% relative to source-only. Replay-only is slightly better than combined alignment (H5 RMSE {q.loc['Hybrid replay','RMSE']:.6f} vs {q.loc['Hybrid both','RMSE']:.6f}); there is no demonstrated alignment-specific gain.'
 body += [p('<b>Interpretation and limits.</b> '+conclusion+' The contribution is an implemented, controlled study of target choice, causal scale normalization, compact forecasting and label-free alignment. Individual ingredients are established. Test periods were inspected in the intensive phase, and H5-mean was introduced after reviewing H1: these are development results, not independent confirmation. Daily peaks remain difficult; no trading-profit or publication-level novelty claim is made. A paired 20-session block bootstrap (2,000 replicates, dates shared across stocks) gives H5 MSE improvement versus Mean 60 a 95% interval of [-0.0443, 0.0453], including zero. Next: confirm on an unseen later period.',9)]
 body += [p('<b>References.</b> Bai, Kolter and Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv:1803.01271, 2018. Gretton, Borgwardt, Rasch, Scholkopf and Smola. A Kernel Two-Sample Test, JMLR 13, 2012 (jmlr.org/papers/v13/gretton12a.html). Sun and Saenko. Deep CORAL, ECCV Workshops, 2016 (arxiv.org/abs/1607.01719). Corsi. A Simple Approximate Long-Memory Model of Realized Volatility, Journal of Financial Econometrics 7(2), 2009 (doi:10.1093/jjfinec/nbp001). Cached Yahoo data via ranaroussi/yfinance; github.com/ranaroussi/yfinance, accessed in original collection. Code: github.com/Ayush20990/ie643-project.',7.5)]
 def footer(c,doc):c.setFont('Helvetica',8);c.setFillColor(INK);c.drawString(40,24,'DeepGen | IE643 novelty assessment | reproducible development results');c.drawRightString(A4[0]-40,24,str(doc.page))
 report=D/'DeepGen_IE643_NoveltyAssessment_report.pdf'
 SimpleDocTemplate(str(report),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=38).build(body,onFirstPage=footer,onLaterPages=footer)
 # A1 poster with official header logo placements and all required sections.
 complete=all(DETAILS[k].strip() for k in ['ayush_email','rishik_email','demo_url'])
 poster=D/('DeepGen_IE643_CourseProject_Poster.pdf' if complete else 'DeepGen_IE643_CourseProject_Poster_DRAFT.pdf')
 c=canvas.Canvas(str(poster),pagesize=A1);W,HH=A1
 c.setFillColor(colors.white);c.rect(0,0,W,HH,fill=1,stroke=0)
 c.drawImage(str(D/'template/asset_0.png'),50,HH-240,width=175,height=175,preserveAspectRatio=True,mask='auto')
 c.drawImage(str(D/'template/course_logo.png'),W-225,HH-240,width=175,height=200,preserveAspectRatio=True,mask='auto')
 c.setFillColor(TEAL);c.setFont('Helvetica-Bold',34);c.drawCentredString(W/2,HH-70,'IE 643 Course Project')
 c.setFillColor(INK);c.setFont('Helvetica-Bold',38);c.drawCentredString(W/2,HH-120,'Forecasting under domain shift')
 c.setFont('Helvetica',28);c.drawCentredString(W/2,HH-161,'From daily spikes to trading-range horizons')
 c.setFont('Helvetica-Bold',27);c.drawCentredString(W/2,HH-204,'Team DeepGen')
 for j,(name,roll,key) in enumerate([('Ayush Patel','23B0037','ayush_email'),('Rishik Yadav','23B2103','rishik_email')]):
  c.setFont('Helvetica',20);c.drawCentredString(W/2,HH-238-j*29,f'{name} | {roll} | '+(DETAILS[key] or 'EMAIL TO BE ADDED'))
 if not complete:
  c.setFillColor(colors.HexColor('#fff1d7'));c.rect(50,HH-321,W-100,35,fill=1,stroke=0)
  c.setFillColor(INK);c.setFont('Helvetica-Bold',16);c.drawCentredString(W/2,HH-309,'REVIEW DRAFT: add member emails and demo-video link before printing')
 margin=55;gap=40;cw=(W-2*margin-gap)/2;xl=margin;xr=margin+cw+gap;yl=yr=HH-355
 yl=heading(c,'Abstract and motivation',xl,yl,cw)
 yl=drawp(c,'We transfer lightweight time-series predictors across weather, transformer and financial domains. Finance motivates a focused extension: forecast next-session range and five-session average range, then test whether feature alignment improves transfer from IT to pharma without target forecast labels.',xl,yl,cw)
 yl=drawp(c,'Daily price movements are noisy. Expected trading range is a useful measure of variability; averaging over five sessions offers an easier but explicitly different forecasting target.',xl,yl,cw)
 yl=heading(c,'Workflow',xl,yl-8,cw)
 for text in ['Intensive: three domains + LiteTCN / hybrid benchmarks','Diagnose weak finance performance and daily spikes','H1 range specialist -> matched baselines + old first step','H5 average-range task -> alignment ablations -> demo']:
  c.setFillColor(PALE);c.roundRect(xl,yl-50,cw,46,6,fill=1,stroke=0);drawp(c,text,xl+12,yl-10,cw-24,18);yl-=59
 yl=heading(c,'Dataset details',xl,yl-8,cw)
 yl=drawp(c,'<b>Weather:</b> Shimla -> Delhi; 7 variables; hourly, 2021-2024; 35,064 rows/city; split 70/10/20.<br/><b>Energy:</b> ETTh1 -> ETTh2; 7 variables; hourly, July 2016-June 2018; 17,420 rows/site; split 60/20/20.<br/><b>Finance:</b> 5 IT -> 5 pharma stocks; 4 inputs; daily trading sessions, January 2012-June 2025; 3,322 rows/company; split 70/10/20.',xl,yl,cw,19)
 yl=heading(c,'Methods and combined architecture',xl,yl-8,cw)
 for text in ['60 past sessions: return | range | gap | log volume','Causal local scaling from the observed window']:
  c.setFillColor(PALE);c.roundRect(xl,yl-49,cw,44,6,fill=1,stroke=0);drawp(c,text,xl+12,yl-11,cw-24,18);yl-=57
  c.setStrokeColor(TEAL);c.line(xl+cw/2,yl+8,xl+cw/2,yl)
 bw=(cw-24)/2
 for x,text in [(xl,'<b>Branch A: causal TCN</b><br/>Width 16; dilations 1 / 2 / 4'),(xl+bw+24,'<b>Branch B: ridge</b><br/>14 multi-scale input features')]:
  c.setFillColor(PALE);c.roundRect(x,yl-88,bw,82,6,fill=1,stroke=0);drawp(c,text,x+10,yl-15,bw-20,18)
 c.setStrokeColor(TEAL);c.line(xl+cw/2,yl+6,xl+bw/2,yl+6);c.line(xl+cw/2,yl+6,xl+bw+24+bw/2,yl+6)
 for x in [xl+bw/2,xl+bw+24+bw/2]:c.line(x,yl+6,x,yl-6);c.line(x,yl-88,x,yl-98)
 yl-=101
 c.line(xl+bw/2,yl+3,xl+bw+24+bw/2,yl+3);c.line(xl+cw/2,yl+3,xl+cw/2,yl-6)
 c.setFillColor(PALE);c.roundRect(xl,yl-49,cw,44,6,fill=1,stroke=0);drawp(c,'Source-selected convex blend -> one range target',xl+12,yl-11,cw-24,18);yl-=57
 yl=drawp(c,'<b>2,449 neural parameters</b> versus 26,888 in the earlier full hybrid; the specialist has a narrower output task. Same scaling and blend for source and adapted comparisons. H1 predicts one daily range; H5 predicts the average of five future daily ranges.',xl,yl,cw,19)
 yl=heading(c,'Novelty assessment: controlled transfer',xl,yl-8,cw)
 yl=drawp(c,'<b>Adaptation loss</b><br/>Source forecast MSE + 0.1 MMD + 0.1 CORAL<br/>+ 0.01 source-weight anchor',xl,yl,cw,22)
 yl=drawp(c,'Only target training <b>inputs</b> enter alignment. Target forecast answers are excluded from fitting and selection. Source validation selects checkpoints and the blend. Compare source-only, replay-only, MMD, CORAL, both, ridge and naive baselines across three seeds.',xl,yl,cw,19)
 yr=heading(c,'Results: finance novelty phase',xr,yr,cw)
 rr=[['Target / model','RMSE','MAE']]
 for h in [1,5]:
  for n in ['Persistence','Mean 60','Hybrid source','Hybrid replay','Hybrid both']:
   rr.append([f'H{h}'+(' mean' if h==5 else '')+' / '+n,f'{qs[h].loc[n,"RMSE"]:.3f}',f'{qs[h].loc[n,"MAE"]:.3f}'])
 t=table(rr,[cw*.58,cw*.21,cw*.21],18);_,ht=t.wrap(cw,2000);t.drawOn(c,xr,yr-ht);yr-=ht+18
 yr=drawp(c,'Range units: 100 ln(High / Low). Three-seed mean. Compare within each target; an easier target is not evidence of better adaptation.',xr,yr,cw,18)
 for file,height in [(paths[5]/'comparison.png',cw*.46),(paths[5]/'forecast.png',cw*.364)]:
  c.drawImage(str(file),xr,yr-height,width=cw,height=height);yr-=height+16
 yr=heading(c,'What improved, and what did not',xr,yr-5,cw)
 yr=drawp(c,f'H5 mean hybrid: <b>{gain(q,"Hybrid both","Persistence"):.1f}%</b> lower RMSE than persistence; <b>{gain(q,"Hybrid both","Mean 60"):.1f}%</b> lower than the 60-session mean. '+conclusion,xr,yr,cw,19)
 yr=heading(c,'Intensive-phase context',xr,yr-5,cw)
 yr=drawp(c,'Earlier H24 hybrid key RMSE: weather <b>1.2836 C</b>, energy <b>3.9762 C</b>. Earlier finance H5 range RMSE: <b>1.0658</b>. Matched H1 first-step RMSE of earlier adapted finance hybrid: <b>'+f'{legacy.loc["Earlier hybrid adapted_model","RMSE"]:.4f}</b>. On the matched H5 mean task, the earlier adapted model has RMSE <b>{legacy5.loc['Earlier hybrid adapted_model','RMSE']:.4f}</b> and MAE <b>{legacy5.loc['Earlier hybrid adapted_model','MAE']:.4f}</b>. The new model improves MAE, but not RMSE.',xr,yr,cw,18)
 yr=heading(c,'Conclusions and future work',xr,yr-5,cw)
 yr=drawp(c,'The range specialist retains a compact TCN and improves on simple baselines. Development results do not establish universal alignment gains or trading profitability. Freeze the protocol, acquire a fresh later period, and evaluate stability using temporal blocks. Compare aggregation horizons before deployment.',xr,yr,cw,18)
 # Full-width footer kept separate from content.
 low=min(yl,yr)
 footer_top=min(500,low-25)
 if footer_top<270:raise RuntimeError(f'Poster content overflow: bottom {low}')
 drawp(c,'<b>References</b><br/>Bai, Kolter and Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv:1803.01271, 2018.<br/>Gretton, Borgwardt, Rasch, Scholkopf and Smola. A Kernel Two-Sample Test, JMLR 13, 2012. jmlr.org/papers/v13/gretton12a.html.<br/>Sun and Saenko. Deep CORAL: Correlation Alignment for Deep Domain Adaptation, ECCV Workshops, 2016. arxiv.org/abs/1607.01719.<br/>Corsi. A Simple Approximate Long-Memory Model of Realized Volatility, Journal of Financial Econometrics 7(2), 2009. doi:10.1093/jjfinec/nbp001.<br/>Data: Open-Meteo historical API; Zhou et al., github.com/zhouhaoyi/ETDataset; ranaroussi/yfinance (cached original project datasets).<br/><b>Acknowledgements:</b> IE643 instructional team; cited dataset and open-source contributors.',xl,footer_top,W-2*margin-330,13)
 qrcode(c,DETAILS['github_url'],W-350,footer_top-140,130);drawp(c,'GitHub repository',W-350,footer_top-145,140,12)
 if DETAILS['demo_url']:qrcode(c,DETAILS['demo_url'],W-190,footer_top-140,130)
 else:
  c.setStrokeColor(TEAL);c.rect(W-190,footer_top-140,130,130);drawp(c,'DEMO LINK<br/>PENDING',W-183,footer_top-55,116,15)
 drawp(c,'Demo video',W-190,footer_top-145,140,12)
 c.setFont('Helvetica',12);c.drawString(55,35,'A1 portrait | DeepGen | 18 November 2026 | Development benchmarks; no independent confirmation')
 c.save()
 # Portable HTML guide, readable without Python.
 guide=f'''<!doctype html><html><head><meta charset="utf-8"><title>DeepGen recovery and novelty guide</title><style>body{{font:18px/1.65 system-ui;color:#173344;background:#f2f6f8;margin:0}}main{{max-width:1050px;margin:35px auto;background:white;padding:48px;border-radius:18px}}h1,h2{{color:#167e76}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:8px;border-bottom:1px solid #cdd; text-align:left}}pre{{background:#eaf3f4;padding:18px;overflow:auto}}img{{max-width:100%}}.note{{background:#fff1d7;padding:16px}}a{{color:#12697a}}</style></head><body><main><h1>DeepGen: recovered project and novelty experiments</h1><p>Restored to <b>D:\\IE643_project</b> from GitHub commit 3b929229be7739819d76702616f4d8410753af68. The recovery includes only files pushed to GitHub. Some later local study-guide edits were absent; this document explains the recovered and new work.</p><h2>1. What changed?</h2><p>The intensive project remains available: weather and energy at 24 hours and rolling next hour; finance at five sessions with four forecast variables. The novelty page adds two specialized finance tasks. H1 predicts tomorrow's range. H5 mean predicts the average of the next five daily ranges. Both keep multivariate inputs and transfer from IT to pharma.</p><p>Range = 100 ln(High / Low). For example, High = 102 and Low = 100 gives approximately 1.98. This describes trading variability, not whether the stock will rise. A five-day average reduces individual-day noise, but its lower absolute error cannot by itself prove a better model.</p><h2>2. Training from scratch</h2><p>Each company's first 70% is training, next 10% validation and final 20% test. Training and validation select source models using IT answers. At every origin, only the previous 60 sessions form the input. Targets never cross split boundaries. A 60-session range mean scales returns, gaps and range; volume is normalized within the observed window.</p><p>The causal TCN has 2,449 weights (the earlier full finance hybrid has 26,888 neural weights; the specialist forecasts a narrower target), a 4-to-16 input projection, three residual convolution layers with dilations 1, 2 and 4, and a linear prediction head. The convolutions see 15 recent positions; the normalization statistics summarize all 60. The head predicts a correction to the last five-session average, and outputs are constrained positive. Source training minimizes MSE in original range units.</p><p>The linear branch uses averages over 1, 5, 20 and 60 sessions, standard deviations, absolute returns and gaps. It has 14 coefficients fitted by ridge regression. Its multi-scale motivation resembles HAR, but the input is daily range rather than the original realized-volatility target. A weight chosen from 0, .25, .5, .75 and 1 combines its prediction with the TCN using source validation only.</p><h2>3. What does adaptation use?</h2><p>Source replay MSE + 0.1 MMD + 0.1 CORAL + 0.01 source-weight penalty. Source replay keeps the model useful on IT. MMD matches feature distributions; CORAL matches feature covariances. The penalty discourages moving too far from the saved source weights. No pharma forecast answers or historical tail regression pairs enter this new procedure.</p><p>Only pharma training input windows are used for alignment. Source validation selects the checkpoint, with the original checkpoint included as a fallback. We also train replay-only, MMD-only and CORAL-only variants. The same source-selected mixture weight is used across all these comparisons.</p><h2>4. Measured results</h2><h3>Next-session range</h3>{q1.round(5).to_html()}<img src="../../results/novelty_finance/comparison.png"><h3>Average range over the next five sessions</h3>{q.round(5).to_html()}<img src="../../results/novelty_finance_mean5/comparison.png"><img src="../../results/novelty_finance_mean5/forecast.png"><p class="note">{conclusion} Never describe baseline improvement or a shorter/easier task as an adaptation improvement. H1 was evaluated before the H5-mean extension was designed. The old test period was already inspected during project development.</p><p>Matched old-hybrid H5 mean: RMSE {legacy5.loc['Earlier hybrid adapted_model','RMSE']:.4f}, MAE {legacy5.loc['Earlier hybrid adapted_model','MAE']:.4f}. It retains better RMSE; the specialist improves MAE. Temporal-block analysis does not establish a reliable H5 gain over the 60-session mean. See results/novelty_finance_mean5/block_uncertainty.csv.</p><h2>5. What did the intensive models contain?</h2><p>The earlier H24/finance hybrid combines ridge, Patch MLP and DLinear branches, source/adapted variants and simple mean forecasts. The weather/energy hourly system combines a small causal LiteTCN, local dynamics and earlier-model first-step forecasts with persistence. Historical target values supplied auxiliary forecasting/calibration answers there. That differs from the new finance alignment experiment, which uses target feature distributions without target prediction labels.</p><p>Retained LiteTCN and boosted LiteTCN were additional experiments; the earlier H24 hybrid remained stronger on key RMSE. See the recovered <a href="../IE643_ASSESSMENT_GUIDE.html">complete intensive guide</a> for the original evidence. The new range specialist does not replace all those models.</p><h2>6. Launch and reproduce</h2><pre>cd /d D:\\IE643_project
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8502</pre><p>Or double-click START_DASHBOARD.bat. Open http://127.0.0.1:8502 and enable <b>Novelty: next-session finance</b> in the sidebar. Select next-session or five-session-average range. Upload one company CSV, or use a built-in example. The app never retrains from the upload. A live forecast has no actual value; historical evaluation displays actuals and metrics.</p><pre>python -m src.novelty_finance --horizon 1
python -m src.novelty_finance --horizon 5
python -m src.novelty_finance_legacy
python -m src.build_novelty_documents</pre><h2>7. Submission requirements</h2><p>Poster: A1 print plus PDF. Report: at most two pages. Deadline: 22 November 2026, 11:59 PM IST. Attend the allocated in-person slot on 23 or 25 November. The poster needs member emails, IITB/course logos, project sections, novelty experiments, results, acknowledgements and GitHub/demo links.</p><p>Edit <b>report/novelty/poster_details.json</b> with the actual emails and demo URL, then rebuild. Until these are supplied the poster is explicitly marked DRAFT. Check links and your assigned presentation slot before submission.</p><h2>8. Speech-ready explanation</h2><blockquote>Our earlier finance model found daily movements difficult. We therefore studied a narrower target: the daily trading range, and then its average over five sessions. We combined a small causal TCN with a multi-scale linear predictor. We also tested MMD and CORAL using target inputs only, with replay-only controls. We report the simpler-task gains separately from adaptation gains, including cases where source validation keeps the original model.</blockquote><h2>9. Likely questions</h2><p><b>Is changing the task itself a novel algorithm?</b> No. It is an experimentally evaluated task reformulation. The model combination, scale handling and controlled adaptation study are project design choices built from established methods.</p><p><b>Is a five-session average a five-step trajectory?</b> No. It is one value summarizing five future daily ranges. The original five-step multivariate model is still available separately.</p><p><b>Does alignment necessarily help?</b> No. It can remove meaningful domain differences, and source-validation selection can reject useful or harmful target changes. This study does not prove unseen target improvement.</p><p><b>Why so much residual error?</b> News-driven spikes are not contained in the past price/volume inputs. Averaging reduces some noise but does not make the future deterministic.</p><p><b>What is missing before print?</b> Confirm actual email addresses and the demo-video link in the poster settings. The current PDF is a review draft if those fields are blank.</p></main></body></html>'''
 (D/'NOVELTY_GUIDE.html').write_text(guide,encoding='utf8')
 # Recreate missing active-file destinations as useful redirects, without pretending exact recovery.
 for name in ['MODEL_ARCHITECTURE_AND_TRAINING.html','IE643_PRESENTATION_MASTERCLASS.html']:
  dest=ROOT/'report'/name
  if not dest.exists():dest.write_text('<!doctype html><meta charset="utf-8"><title>Recovered project guide</title><h1>Recovered project guide</h1><p>The previous local version was not in the GitHub snapshot. Open the reconstructed <a href="novelty/NOVELTY_GUIDE.html">architecture, training and novelty guide</a> or the recovered <a href="IE643_ASSESSMENT_GUIDE.html">intensive assessment guide</a>.</p>',encoding='utf8')
 validation={'report_pages':len(pymupdf.open(report)),'poster_pages':len(pymupdf.open(poster)),'poster_size_points':list(pymupdf.open(poster)[0].rect),'poster_print_ready':complete,'missing_fields':[k for k in ['ayush_email','rishik_email','demo_url'] if not DETAILS[k]],'poster':str(poster)}
 (D/'document_validation.json').write_text(json.dumps(validation,indent=2))
 for path in [report,poster]:
  doc=pymupdf.open(path)
  for i,page in enumerate(doc):page.get_pixmap(matrix=pymupdf.Matrix(.6,.6) if path==poster else pymupdf.Matrix(1.3,1.3)).save(str(ROOT/f'tmp/novelty/{path.stem}_{i+1}.png'))
 print(json.dumps(validation,indent=2))
 if validation['report_pages']!=2:raise RuntimeError('Report must be exactly two pages')

if __name__=='__main__':main()
