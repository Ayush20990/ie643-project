"""Build offline dashboard and exact numerical report tables from saved results."""
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from .build_assessment_guide import build_tables
from .config import ROOT,RESULTS_DIR
from .submission import benchmark_table


def tex_escape(s):
    replacements={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#","_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return ''.join(replacements.get(c,c) for c in str(s)).replace(r'\_',r'\_\allowbreak{}').replace('→',r'$\rightarrow$').replace('—','-').replace('°',r'$^{\circ}$')


def latex_table(markdown, caption, label):
    rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in markdown.splitlines()]
    headings,body=rows[0],rows[2:]
    n=len(headings)
    ratios={3:[.58,.21,.21],4:[.12,.5,.19,.19],5:[.32,.17,.17,.17,.17],6:[.12,.2,.19,.2,.145,.145],7:[.1,.22,.12,.16,.14,.13,.13]}.get(n,[1/n]*n)
    # Space for inter-column padding is reserved explicitly.
    columns='@{}'+''.join('p{'+f'{v*.91:.4f}'+r'\textwidth}' for v in ratios)+'@{}'
    row=lambda cells:' & '.join(tex_escape(c) for c in cells)+r' \\'+'\n'
    head=row(headings)
    return ('{\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.17}\n'
            +r'\begin{longtable}{'+columns+'}\n'+r'\caption{'+tex_escape(caption)+r'}\label{'+label+r'}\\'+'\n'
            +r'\toprule'+'\n'+head+r'\midrule\endfirsthead'+'\n'+head+r'\midrule\endhead'+'\n'
            +r'\midrule\multicolumn{'+str(n)+r'}{r}{Continued on next page}\\\endfoot'+'\n'+r'\bottomrule\endlastfoot'+'\n'
            +''.join(row(r) for r in body)+r'\end{longtable}'+'\n}\n')


def main():
    out=ROOT/'report'/'intensive';out.mkdir(parents=True,exist_ok=True)
    tables=build_tables()
    captions={
        'ORIGINAL_TABLE':'Original source/adaptation campaign, target MSE, three seeds.',
        'V2_TABLE':'Original saved CARE-DA v2 comparison: fixed seed 0, not a three-seed v2 mean.',
        'DEPTH_TABLE':'Matched source-training architecture sweep evaluated on the target, three seeds. No target Y is used to choose the source architecture.',
        'LONG_CLIMATE':'Climate, single-origin H=24: target development results. Key RMSE in degrees Celsius.',
        'LONG_ENERGY':'Energy, single-origin H=24: target development results. Key RMSE is oil-temperature degrees Celsius.',
        'LONG_FINANCE':'Finance, H=5 trading days: target development results. Key RMSE is 100 log(high/low), not price RMSE.',
        'ROLLING_HEADLINE':'Final peak-calibrated H=1 hybrid, target temperature RMSE, three-seed means. Comparisons share the same horizon.',
        'ROLLING_COMPARISON':'Matched next-hour comparisons. The original checkpoint is fixed seed 0; other fitted families average three seeds.',
        'PEAK_TABLE':'Errors over non-overlapping 24-hour episodes of successive H=1 forecasts; the last incomplete episode is omitted.',
        'BOOST_ABLATION':'H=24 added-tree control with backbone, decoder and mixture held fixed; this is not a separately reselected source-only policy.',
        'EFFICIENCY':'Deployment CPU efficiency. KB includes active native trees; neural counts exclude native tree leaves and coefficient buffers.',
        'SOURCE_RETENTION':'Both evaluated domains for final H=1 source-only and adapted models, three-seed means.'}
    for key,table in tables.items():
        (out/(key.lower()+'.tex')).write_text(latex_table(table,captions[key],'tab:'+key.lower()),encoding='utf-8')
    # Complete original ablation and labeled references, rather than favorable subsets.
    rows=[]
    for ds in ('climate','energy','finance'):
        f=pd.read_csv(RESULTS_DIR/ds/'metrics_all.csv');f=f[f.eval_domain=='target']
        for method,g in f.groupby('method'):
            rows.append({'Dataset':ds,'Original campaign method':method,'MSE':g.MSE.mean(),'Key RMSE':g.key_RMSE.mean()})
    from .build_assessment_guide import md_table
    (out/'original_complete.tex').write_text(latex_table(md_table(rows),'Complete original campaign including negative ablations and explicitly label-using reference fits.','tab:original_complete'),encoding='utf-8')
    rows=[]
    for family in ('peak_litetcn','peak_calibrated','residual_litetcn','progressive_litetcn','coupled_litetcn','coupled_transfer','event_boost','next_hour','next_hour_hybrid','peak_next_hour'):
        by={ds:pd.read_csv(RESULTS_DIR/family/ds/'metrics_all.csv').query("domain=='target'").groupby('method')[['MSE','key_RMSE']].mean() for ds in ('climate','energy')}
        for method in sorted(set(by['climate'].index)|set(by['energy'].index)):
            if method in ('seasonal_naive','persistence'):continue
            row={'Recorded family / method':family+' / '+method}
            for ds in ('climate','energy'):
                for key in ('MSE','key_RMSE'):row[ds+' '+key]=by[ds].loc[method,key] if method in by[ds].index else '-'
            rows.append(row)
    (out/'additional_controls.tex').write_text(latex_table(md_table(rows),'Additional recorded controls. Family names indicate experiment stage; screen controls can be seed 0, whereas selected policies average three seeds. H=1 families begin at next_hour.','tab:additional_controls'),encoding='utf-8')

    dashboard=ROOT/'dashboard';dashboard.mkdir(exist_ok=True)
    sections=[];first=True
    for ds in ('climate','energy','finance'):
        for h in ([1,24] if ds!='finance' else [5]):
            f=benchmark_table(ds,h);target=f[f.domain=='target'].copy()
            target['model']=target.family+' / '+target.method
            target['model']=target['model'].str.replace('Peak-calibrated next-hour hybrid','Peak-calibrated hybrid',regex=False).str.replace('Next-hour LiteTCN + local dynamics','TCN + dynamics',regex=False).str.replace('Earlier hybrid, first-hour output','Earlier hybrid (first hour)',regex=False)
            charts=[]
            for key,title in (('key_RMSE','Key-variable RMSE'),('MSE','Multivariate standardized MSE')):
                unit='Temperature RMSE (degC)' if ds!='finance' else '100 log(high/low) RMSE'
                fig=px.bar(target,x=key,y='model',orientation='h',color='method',title=title,labels={key:unit if key=='key_RMSE' else 'Standardized multivariate MSE'},
                           color_discrete_sequence=['#2678bf','#d7803e','#70a99b','#a0aab5','#9b8ac6'])
                fig.update_layout(template='plotly_white',height=max(360,len(target)*34),margin=dict(l=10,r=10,t=55,b=20),yaxis_title='',font=dict(family='Arial',size=12))
                charts.append(pio.to_html(fig,full_html=False,include_plotlyjs=True if first else False,config={'responsive':True,'displaylogo':False}));first=False
            event=''
            if h==1:
                p=pd.read_csv(RESULTS_DIR/'forecast_horizons'/f'{ds}_next_hour_peak_comparison.csv')
                event='<h3>Rolling daily peak / trough evidence</h3>'+p.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
            sections.append(f'<section data-ds="{ds}" data-h="{h}"><h2>{ds.title()} · H={h} '+('trading days' if ds=='finance' else 'hour' if h==1 else 'hours')+'</h2>'+
                            '<p>Target-domain bars; full source and target table below. Lower error is better.</p>'+''.join(charts)+
                            '<div class="scroll">'+f.to_html(index=False,float_format=lambda v:f'{v:.5f}',border=0)+'</div><div class="scroll">'+event+'</div></section>')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepGen · IE643 results dashboard</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f2f6f9;color:#193247;font:16px/1.6 system-ui}header{background:#14354a;color:white;padding:30px max(24px,calc((100vw - 1180px)/2))}h1{margin:5px 0;font-size:34px}main{max-width:1240px;margin:auto;padding:25px}nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{padding:10px 16px;border:1px solid #becdda;border-radius:5px;background:white;font:inherit}section{background:white;padding:25px;margin-top:25px;border-radius:8px;display:none}section.active{display:block}.note{border-left:4px solid #3e87af;padding:15px 20px;background:#e5f0f7}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;min-width:670px}td,th{padding:10px;border-bottom:1px solid #dae3e9;text-align:left}th{background:#edf4f8}footer{margin-top:30px;color:#637a89}a{color:#176696}@media(max-width:650px){main{padding:14px}section{padding:12px}h1{font-size:26px}}</style></head><body>
    <header><small>TEAM DEEPGEN · IE643 COURSE PROJECT</small><h1>Forecasting across domains</h1><p>Ayush Patel · 23B0037 &nbsp; | &nbsp; Rishik Yadav · 23B2103</p></header><main>
    <nav><label>Dataset <select id="ds"><option value="climate">Climate</option><option value="energy">Energy</option><option value="finance">Finance</option></select></label><label>Task <select id="h"></select></label><button onclick="window.print()">Print</button></nav>
    <p class="note">H=1 is rolling next-hour forecasting with observations refreshed every hour. H=24 predicts a full day from one origin. Finance uses five trading days. Compare models within the same dataset and horizon.</p>
    '''+''.join(sections)+'''
    <h2>Interpretation and limitations</h2><p>The final peak-calibrated H=1 hybrid achieves 0.6530°C climate and 0.9151°C energy temperature RMSE, compared with persistence 1.2692°C and 1.3334°C. Adaptation itself reduces the matching source-only model's RMSE by 3.84% and 27.68%. These gains must not be described as H=24 model improvements.</p><p>Energy persistence still has lower trough MAE. Earlier models remain stronger on some metrics. Finance forecasts engineered features, not stock prices or trading returns; the strict H=5 model improves overall MSE about 1.60% over input mean. These are reused development-test results, not independent confirmation.</p>
    <footer>All charts and Plotly are embedded for offline use. Tables come from saved experiment CSVs. Three-seed means unless a fixed seed-0 reference is named. Source-only versus adapted comparisons use the same evaluated domain.</footer>
    </main><script>const ds=document.getElementById('ds'),h=document.getElementById('h');function show(){document.querySelectorAll('section[data-ds]').forEach(s=>s.classList.toggle('active',s.dataset.ds===ds.value&&s.dataset.h===h.value));document.querySelectorAll('section.active .plotly-graph-div').forEach(e=>Plotly.Plots.resize(e));}function change(){h.innerHTML=(ds.value==='finance'?[[5,'5 trading days']]:[[1,'Rolling next hour'],[24,'Single-origin 24 hours']]).map(x=>'<option value="'+x[0]+'">'+x[1]+'</option>').join('');show()}ds.onchange=change;h.onchange=show;change();</script></body></html>'''
    (dashboard/'IE643_Results_Dashboard.html').write_text(doc,encoding='utf-8')
    used=[p for folder in (RESULTS_DIR/'forecast_horizons',RESULTS_DIR/'peak_next_hour') for p in folder.rglob('*.csv')]
    (out/'numerical_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},indent=2),encoding='utf-8')
    print(dashboard/'IE643_Results_Dashboard.html')


if __name__=='__main__':main()
