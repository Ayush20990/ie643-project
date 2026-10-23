// Local fallback: the presentation session/runtime loader is unavailable here.
// Reuse the existing repository dependency and original midterm artwork/fonts.
const fs=require('fs'),path=require('path'),PptxGenJS=require('pptxgenjs');
const root=path.resolve(__dirname,'..'),out=path.join(__dirname,'intensive');
const E=JSON.parse(fs.readFileSync(path.join(out,'evidence.json'),'utf8'));
const pptx=new PptxGenJS();pptx.layout='LAYOUT_WIDE';
pptx.author='Ayush Patel; Rishik Yadav';pptx.company='DeepGen';pptx.lang='en-IN';
pptx.subject='IE643 intensive assessment';pptx.title='DeepGen - Domain adaptation for multivariate forecasting';
pptx.theme={headFontFace:'Cambria',bodyFontFace:'Calibri',lang:'en-IN'};
const C={ink:'1B2B31',basalt:'16232A',teal:'2F7F73',tealLt:'6FB3A6',saffron:'D98A1E',
 slate:'4F5F65',muted:'7C8B90',line:'CFDAD8',panel:'F1F6F5',glacier:'E6F0EE',warm:'FBEBD3',
 white:'FFFFFF',rose:'B5473A',purple:'8773A9'};
const A=f=>path.join(__dirname,'assets',f),notes=[];
const f=(v,d=4)=>Number(v).toFixed(d),pct=v=>Number(v).toFixed(2)+'%';
function text(s,t,x,y,w,h,size=20,o={}){s.addText(t,{x,y,w,h,fontFace:'Calibri',fontSize:size,color:C.ink,margin:0,
 breakLine:false,valign:'top',paraSpaceAfterPt:5,...o});}
function panel(s,x,y,w,h,color=C.panel){s.addShape(pptx.ShapeType.rect,{x,y,w,h,fill:{color},line:{color,transparency:100}});}
function line(s,x,y,w,color=C.line){s.addShape(pptx.ShapeType.line,{x,y,w,h:0,line:{color,width:1}});}
function base(title,section,body,time,source='report/IE643_ASSESSMENT_GUIDE.md',dark=false){
 const s=pptx.addSlide(),n=pptx._slides.length;s.background={path:A(dark?'bg_title.png':'bg_light_corner.png')};
 if(!dark){text(s,section.toUpperCase(),.6,.32,11.8,.25,10.5,{color:C.teal,bold:true,charSpacing:2.5});
 text(s,title,.6,.7,12.1,.67,30,{fontFace:'Cambria',bold:true});
 text(s,'IE643 Course Project · DeepGen',.6,7.05,10,.2,10,{color:C.muted});
 text(s,n<=18?`${n} / 18`:'References',11.5,7.05,1.2,.2,10,{color:C.muted,align:'right'});}
 const note=`Planned time: ${time}\n${body}\nEvidence: ${source}\nTables: three-seed target means unless named seed 0. Forecast plots: deployment seed 0. H=24 and rolling H=1 are distinct tasks.`;
 s.addNotes(note);notes.push(`## ${n}. ${title} (${time})\n\n${body}\n\nEvidence: ${source}\n`);return s;
}
function caption(s,t,y=6.63,color=C.slate){text(s,t,.65,y,12.0,.35,14,{color});}
function table(s,headers,rows,x,y,w,widths,size=17,rowH=.54){
 const styled=[headers.map(t=>({text:t,options:{bold:true,color:C.white,fill:C.basalt}})),
 ...rows.map((r,i)=>r.map((v,j)=>({text:String(v),options:{fill:i%2?C.white:C.panel,color:j?C.ink:C.teal,bold:j===0}})))];
 s.addTable(styled,{x,y,w,colW:widths,rowH,fontFace:'Calibri',fontSize:size,color:C.ink,
 margin:[.1,.12,.08,.12],border:{color:C.line,pt:.5},autoPage:false});
}
function metric(s,value,label,x,y,w=3,color=C.teal){text(s,value,x,y,w,.67,36,{fontFace:'Cambria',bold:true,color});text(s,label,x,y+.75,w,.65,17,{color:C.slate});}
function item(s,num,title,body,x,y,w=5.7){text(s,num,x,y,.65,.55,28,{fontFace:'Cambria',bold:true,color:C.teal});
 text(s,title,x+.85,y+.02,w-.85,.42,21,{bold:true});text(s,body,x+.85,y+.58,w-.9,.83,17,{color:C.slate});}
function contain(s,file,x,y,w,h){const b=fs.readFileSync(file),iw=b.readUInt32BE(16),ih=b.readUInt32BE(20),scale=Math.min(w/iw,h/ih);
 s.addImage({path:file,x:x+(w-iw*scale)/2,y:y+(h-ih*scale)/2,w:iw*scale,h:ih*scale});}
function flow(s,items,y=2.0,h=1.35){const gap=.3,w=(12.05-gap*(items.length-1))/items.length;
 items.forEach((a,i)=>{const x=.65+i*(w+gap);panel(s,x,y,w,h,i%2?C.glacier:C.panel);
 text(s,a[0],x+.15,y+.2,w-.3,.45,20,{bold:true,color:C.teal});text(s,a[1],x+.15,y+.74,w-.3,h-.8,16,{color:C.slate});
 if(i<items.length-1)s.addShape(pptx.ShapeType.line,{x:x+w,y:y+h/2,w:gap,h:0,line:{color:C.slate,width:1.3,endArrowType:'triangle'}});});}
function chart(s,series,x,y,w,h,options={}){const values=series.flatMap(a=>a.values),low=Math.min(...values),high=Math.max(...values),pad=Math.max((high-low)*.12,.5);
 s.addChart(pptx.ChartType.line,series,{x,y,w,h,
 chartColors:[C.ink,C.purple,C.saffron,C.teal],lineSize:2.2,lineDataSymbol:'none',
 showLegend:true,legendPos:'b',legendFontFace:'Calibri',legendFontSize:12,
 dataLabelFontFace:'Calibri',titleFontFace:'Cambria',catAxisTitleFontFace:'Calibri',valAxisTitleFontFace:'Calibri',
 catAxisLabelFontFace:'Calibri',catAxisLabelFontSize:12,catAxisLabelColor:C.slate,
 valAxisLabelFontFace:'Calibri',valAxisLabelFontSize:12,valAxisLabelColor:C.slate,
 catAxisLabelFrequency:4,catAxisLineColor:C.line,valAxisLineColor:C.line,
 valGridLine:{color:'E4EBE9',width:.6},catGridLine:{style:'none'},
 showValue:false,showTitle:false,showBorder:false,showMarker:false,showLine:true,
 valAxisNumFmt:'0',showValAxisTitle:false,showShadow:false,
 valAxisMajorUnit:5,
 valAxisMinVal:Math.floor(low-pad),valAxisMaxVal:Math.ceil(high+pad),
 chartArea:{fill:{color:C.white}},plotArea:{fill:{color:C.white}},...options});}
function series(p,role){return role==='H24'?
 [{name:'Actual',labels:p.hours,values:p.actual},{name:'Retained LiteTCN',labels:p.hours,values:p.retained},
 {name:'Boosted LiteTCN',labels:p.hours,values:p.adapted},{name:'Earlier hybrid',labels:p.hours,values:p.earlier}]:
 [{name:'Actual',labels:p.hours,values:p.actual},{name:'Source-only',labels:p.hours,values:p.source},{name:'Adapted',labels:p.hours,values:p.adapted}];}

let s=base('Domain adaptation for multivariate forecasting','',
 'Introduce DeepGen and the IE643 task. Both members researched methods and ran experiments. Ayush developed the dashboard/app; Rishik prepared the report and presentation.','0:15',undefined,true);
panel(s,0,0,8.9,7.5,C.basalt);
text(s,'IE643 COURSE PROJECT',.72,.72,8,.3,12,{bold:true,color:C.tealLt,charSpacing:3});
text(s,'Domain adaptation\nfor multivariate\nforecasting',.72,1.4,8.1,2.6,42,{fontFace:'Cambria',bold:true,color:C.white});
text(s,'Lightweight models, observed-input adaptation\nand peak-sensitive evaluation',.76,4.15,7.5,.9,23,{color:C.glacier});
text(s,'DeepGen',.76,5.48,7,.45,25,{bold:true,color:C.white});
text(s,'Ayush Patel · 23B0037\nRishik Yadav · 23B2103',.76,6.08,7,.72,18,{color:C.glacier});
text(s,'INTENSIVE\nASSESSMENT',9.55,5.2,3.1,.9,21,{bold:true,color:C.white});
text(s,'Revised 23 October 2026',9.55,6.3,3.1,.35,16,{color:C.tealLt});

s=base('Outline','The assessment','Follow the problem and the midterm model, then explain the changes. Separate the original day-ahead evidence and added next-hour evidence. Finish with app, limitations and next assessment plan.','0:15');
item(s,'01','Problem and domain pairs','Three datasets with separate source and target domains',.7,1.8);
item(s,'02','Work before and after the review','LiteTCN, depth controls and the reason for the hybrid',6.8,1.8);
item(s,'03','Adaptation and evaluation','Observed-input pretexts, source guards and frozen inference',.7,4.05);
item(s,'04','Results and demonstration','Day-ahead, hourly and finance forecasts, followed by the CSV app',6.8,4.05);

s=base('Forecasting under domain shift','Problem understanding',
 'Source supplies histories and forecast labels. Adaptation uses target histories only. The unseen future of the original target forecasting task stays excluded. Historical values inside X supply self-supervised answers.','0:30','src/config.py; src/data.py; src/strict_models.py');
panel(s,.65,1.8,5.8,3.4,C.glacier);panel(s,6.75,1.8,5.9,3.4,C.warm);
text(s,'DOMAIN 1',.9,2.02,5,.3,13,{color:C.teal,bold:true,charSpacing:2});
text(s,'Source training',.9,2.52,5.1,.6,29,{fontFace:'Cambria',bold:true});
text(s,'Observed histories X\nFuture labels Y\nSupervised forecasting loss',.9,3.45,5.1,1.35,23);
text(s,'DOMAIN 2',7.0,2.02,5,.3,13,{color:C.saffron,bold:true,charSpacing:2});
text(s,'Target adaptation',7.0,2.52,5.1,.6,29,{fontFace:'Cambria',bold:true});
text(s,'Observed histories X only\nWithheld historical values for practice\nUnseen forecast Y excluded',7.0,3.45,5.2,1.35,22);
text(s,'The shift can change levels, dynamics and relationships between variables.',.8,5.8,11.8,.75,24,{bold:true});

s=base('Project workflow','Methodology',
 'Collect domain pairs and split chronologically. Train on source labels, adapt only from observed target inputs with source safeguards, freeze before evaluation and deploy through CSV inference.','0:25','src/data.py; src/nowcast_train.py; src/peak_nowcast_train.py; src/submission.py');
flow(s,[['Collect','Three domain pairs'],['Prepare','Chronological splits'],['Train','Source X and Y'],['Adapt','Observed target X'],['Evaluate','Frozen forecasts']],2.1,1.6);
text(s,'One experiment record connects data, checkpoint and result',.8,4.32,11.8,.7,26,{fontFace:'Cambria',bold:true});
text(s,'Saved selection manifests and checkpoints\nForecast CSVs, event metrics and qualitative plots\nA single interface for both domains',.85,5.2,11.4,1.3,21,{color:C.slate});

s=base('Before the review: LiteTCN and CARE-DA','Midterm baseline',
 'Original model: RevIN, strided convolutions, a linear path and reconstruction head. CARE combines supervised forecasting, alignment, self-supervision and an anchor. These saved v2 results are fixed seed 0, not three-seed means.','0:35','src/model.py; src/train.py; results/*/care_v2_compare.csv');
flow(s,[['RevIN','Window normalization'],['LiteTCN','Temporal encoder'],['Forecast head','Multivariate output']],1.65,1.3);
text(s,'Linear temporal path and reconstruction support the encoder',.8,3.23,11.8,.45,20,{color:C.slate});
table(s,['Original saved comparison','Source MSE','CARE-DA v2 MSE'],[['Climate · H=24','0.2512','0.2357'],['Energy · H=24','0.1691','0.1559'],['Finance · H=5','0.5622','0.5520']],.75,4.05,11.85,[5.85,3,3],19,.52);
caption(s,'Fixed seed 0. Earlier campaign results, separate from the final headline.');

s=base('After the review: depth alone was insufficient','Experiments and transition',
 'Deeper TCNs were tested alongside linear and Patch MLP controls. TCN revisions and geometry-aware boosting gave mixed or modest day-ahead gains. The final next-hour hybrid keeps a causal LiteTCN and complementary experts. H=1 is an explicitly added task.','0:30','results/improved; results/peak_litetcn; results/event_boost; results/peak_next_hour');
const changes=[['01','Depth and architecture','TCN depths 3 / 5 / 7, Patch MLP and DLinear controls'],['02','Strict adaptation','Input-tail pretexts and guarded source/adapted mixtures'],['03','Day-ahead peak experiments','Residual, gradual-unfreezing, driver and boosted variants'],['04','Next-hour hybrid','Small LiteTCN, local dynamics and observed-event calibration']];
changes.forEach((a,i)=>{let y=1.68+i*1.13;text(s,a[0],.75,y,.65,.55,29,{fontFace:'Cambria',bold:true,color:i%2?C.saffron:C.teal});text(s,a[1],1.65,y+.02,10.8,.4,22,{bold:true});text(s,a[2],1.65,y+.52,10.8,.43,18,{color:C.slate});});
caption(s,'LiteTCN remains in the final system. The final headline belongs to the full hybrid.',6.63,C.teal);

s=base('Three datasets with distinct domains','Data and acquisition',
 'Weather comes from Open-Meteo. ETT contains transformer loads and oil temperature. Finance pools company series by sector and predicts engineered features. All configured variables are forecast.','0:40','src/config.py; src/data_download.py; https://open-meteo.com/en/docs/historical-weather-api; https://github.com/zhouhaoyi/ETDataset');
table(s,['Domain pair','Rows / domain','Forecast variables','Lookback / horizon'],[['Shimla to Delhi','35,064','7 · hourly weather','168 h / 24 h or 1 h'],['ETTh1 to ETTh2','17,420','7 · load and oil temperature','168 h / 24 h or 1 h'],['IT to pharma','16,610','4 · daily financial features','60 days / 5 days']],.65,1.72,12.05,[3.0,2.0,3.6,3.45],18,.72);
text(s,'What the plots emphasize',.8,5.02,11.8,.5,24,{fontFace:'Cambria',bold:true});
text(s,'Weather: air temperature. Energy: transformer oil temperature.\nFinance: high–low range, alongside return, opening gap and log volume.',.8,5.73,11.8,.85,19,{color:C.slate});

s=base('Observed history supplies the adaptation task','Preprocessing and leakage boundary',
 'Weather/finance splits 70/10/20, ETT 60/20/20. Scalers use each domain training period. All hours 1–168 are observed. The inner predictor hides and predicts the tail. Actual future Y stays excluded. For H=1, use 1–167 to predict observed 168; actual inference predicts 169. App inference uses fixed weights.','0:45','src/data.py; src/strict_models.py; src/peak_nowcast_train.py');
text(s,'Example: a 24-hour pretext inside seven observed days',.75,1.62,11.8,.5,23,{bold:true});
table(s,['Role','Hours','Use'],[['Inner predictor input','1–144','First six observed days'],['Practice answers','145–168','Seventh day, already observed'],['Actual input X','1–168','All seven days at forecast time'],['Actual future Y','169–192','Excluded from adaptation']],.75,2.28,11.85,[3.7,2.05,6.1],19,.58);
text(s,'Chronological splits and training-period scalers',.85,5.65,11.5,.45,22,{bold:true,color:C.teal});
caption(s,'H=1 uses a one-hour pretext. Rolling forecasts refresh observations; model weights stay fixed.');

s=base('LiteTCN within the final next-hour hybrid','Forecasting model',
 'The new causal TCN has 8,903 parameters. It predicts increments for seven variables from 48 observed hours plus clock features. Local dynamics is complementary. Five outer experts are source/adapted new hybrids, source/adapted earlier-hybrid first-hour outputs and persistence. Mixture weights depend on variable and causal regime.','0:45','src/nowcast_tcn.py; src/nowcast_hybrid.py; src/peak_nowcast.py');
panel(s,.7,1.7,5.6,3.48,C.glacier);text(s,'Small causal LiteTCN',.95,1.98,5.1,.5,26,{fontFace:'Cambria',bold:true});
text(s,'48 observed hours\nWidth 32, dilations 1 / 2 / 4\nNext-hour increments for all variables',.95,2.87,5.0,1.55,21);
text(s,'Zero head begins at persistence',.95,4.59,5.0,.4,17,{color:C.teal,bold:true});
text(s,'Five experts, one next-hour forecast',6.75,1.96,5.85,.7,25,{fontFace:'Cambria',bold:true});
text(s,'New hybrid: source and adapted\nEarlier hybrid: source and adapted first hour\nPersistence: latest observed value',6.75,2.96,5.7,1.65,21);
text(s,'Past slopes select the regime. Convex weights combine the experts.',6.75,4.75,5.6,1.1,20,{color:C.slate});
caption(s,'Every expert predicts the same next hour. The mixture does not average different horizons.');

s=base('Adaptation objective and safeguards','Loss design',
 'TCN adaptation: target-input forecasting + 0.5 source replay + 0.001 relative anchor. Event calibration uses daily episodes within observed inputs. Extrema are loss labels only, never regime features. SLSQP fits convex weights. Source point guard 10%, target-input guard 5%, minimum 64 examples per regime and source fallback.','0:55','src/nowcast_train.py; src/peak_nowcast_train.py');
panel(s,.7,1.72,11.9,1.64,C.basalt);text(s,'Event loss',1.0,1.95,11.2,.4,20,{color:C.tealLt,bold:true});
text(s,'Point MSE + 2 × peak/trough MSE + 0.1 × ramp MSE',1.0,2.57,11.1,.5,25,{fontFace:'Cambria',bold:true,color:C.white});
item(s,'01','Historical target episodes','Every inner origin sees only its preceding observations',.85,4.0,5.6);
item(s,'02','Source protection','Replay, an anchor, error guards and source fallback',6.85,4.0,5.6);
caption(s,'Fit mixture weights before test inference. Future target labels never enter the adaptation loss.');

s=base('Experimental setup and metrics','Evaluation protocol',
 'Three seeds measure training randomness. Deployment is seed 0. TCN budgets are sampled rounds, not full epochs. Compare matching source-only/adapted checkpoints. Development test periods were inspected during iterations, so they are not independent confirmation.','0:30','results/next_hour/*/manifest.json; results/peak_next_hour/*/manifest.json; results/forecast_horizons/efficiency.csv');
metric(s,'3 seeds','0 / 1 / 2\nDeployment: seed 0',.85,1.85,3.6);metric(s,'28 + 12','Sampled source / adaptation rounds\nUp to 4,096 windows per round',4.8,1.85,3.9,C.saffron);metric(s,'CPU','PyTorch 2.6.0\nRyzen 5 5600H',9.0,1.85,3.3);
line(s,.8,4.15,11.7);
text(s,'Multivariate MSE',.85,4.65,3.6,.5,22,{bold:true});text(s,'Standardized error across\nall forecast variables',.85,5.35,3.5,1,19,{color:C.slate});
text(s,'Physical-unit RMSE',4.8,4.65,3.8,.5,22,{bold:true});text(s,'Temperature in °C\nFinancial range in log-percent units',4.8,5.35,3.8,1,19,{color:C.slate});
text(s,'Event and efficiency',9.0,4.65,3.7,.5,22,{bold:true});text(s,'Peak / trough / ramp errors\nParameters, size and latency',9.0,5.35,3.5,1,19,{color:C.slate});
caption(s,'Compare within the same dataset and horizon. These are reused development-test periods.');

s=base('Original-horizon results','Results · 24 hours and finance',
 'Target three-seed means. Climate earlier hybrid: shorter-context strict ensemble. Energy earlier hybrid: strict ensemble as temperature-focused reference. Guarded DLinear has lower global MSE 0.144778 but oil RMSE 4.074870. Retained vs boosted is a whole-system comparison, not adaptation-only. Finance stays H=5. Right side compares global MSE with seasonal naive for hourly datasets and input mean for finance.','0:50','presentation/intensive/evidence.json; results/strict_context; results/strict_refined; results/forecast_horizons/twenty_four_hour_headline.csv');
let rows=[];for(const ds of ['climate','energy']){const r=E.datasets[ds],name=ds==='climate'?'Climate':'Energy';rows.push([`${name} · earlier hybrid`,f(r.earlier_hybrid.MSE),f(r.earlier_hybrid.key_RMSE)]);rows.push([`${name} · retained LiteTCN`,f(r.daily.retained_MSE),f(r.daily.retained_key_RMSE)]);rows.push([`${name} · boosted LiteTCN`,f(r.daily.adapted_MSE),f(r.daily.adapted_key_RMSE)]);}
rows.push(['Finance · strict hybrid (H=5)',f(E.finance.strict_adapted.MSE),f(E.finance.strict_adapted.key_RMSE)]);
table(s,['Target model / policy','Global MSE','Key RMSE'],rows,.65,1.65,8.65,[5.25,1.65,1.75],17,.55);
text(s,'MSE reduction\nover simple baseline',9.7,1.7,2.9,.8,19,{bold:true});
for(const [i,ds] of ['climate','energy'].entries()){const r=E.datasets[ds];metric(s,pct(100*(1-r.earlier_hybrid.MSE/r.seasonal_naive.MSE)),ds==='climate'?'Climate vs seasonal naive':'Energy vs seasonal naive',9.7,2.7+i*1.4,2.9,i?C.saffron:C.teal);}
metric(s,pct(100*(1-E.finance.strict_adapted.MSE/E.finance.input_mean.MSE)),'Finance vs input mean',9.7,5.35,2.9);
caption(s,'Key RMSE: temperature °C for weather/energy; 100 log(high/low) for finance. H=24 except finance H=5.');

s=base('24-hour forecasts from a single origin','Results · day-ahead trajectories',
 'First fixed target origin, seed 0, without visual performance filtering. Each curve is issued at once without updates inside the 24-hour horizon. Earlier temperature-focused hybrid remains a reference. Boosting lowers retained-TCN temperature RMSE only 1.11% climate and 2.01% energy and does not beat the earlier hybrid. Right-side values are three-seed whole-test RMSE, not the plotted-window error. Peak timing/amplitude errors remain.','0:45','results/event_boost/*/forecast_examples.npz; results/strict_context/climate/forecast_examples.npz; results/strict_refined/energy/forecast_examples.npz; presentation/intensive/evidence.json');
for(const [i,ds] of ['climate','energy'].entries()){const y=1.61+i*2.37,r=E.datasets[ds];text(s,ds==='climate'?'Delhi air temperature (°C)':'ETTh2 oil temperature (°C)',.85,y,8.45,.35,19,{bold:true});chart(s,series(r.H24,'H24'),.65,y+.39,8.65,1.94,{catAxisLabelFrequency:4});text(s,ds==='climate'?'CLIMATE · H=24':'ENERGY · H=24',9.6,y+.1,2.9,.35,12,{color:C.teal,bold:true,charSpacing:1.5});text(s,`${f(r.daily.retained_key_RMSE,3)} → ${f(r.daily.adapted_key_RMSE,3)} °C`,9.6,y+.67,2.98,.45,22,{fontFace:'Cambria',bold:true});text(s,`${pct(r.daily.key_RMSE_gain_vs_retained_pct)} lower RMSE\nBoosted vs retained LiteTCN`,9.6,y+1.28,2.9,.8,17,{color:C.slate});}
caption(s,'Forecast hours 1–24. Fixed first origin, seed 0. Peak timing and amplitude errors remain.');

s=base('Finance: modest gains over a strong baseline','Results · IT to pharma · H=5 trading days',
 'Finance needs explicit qualitative and quantitative coverage. All four engineered variables are forecast over five trading days from 60 observed trading days. Two illustrative outputs are shown: high-low range and daily log return, both from the first fixed saved target origin, deployment seed 0. The example is not selected by error. Source/adapted curves nearly overlap. Flat return forecasts miss the observed sign changes, and range amplitude is biased in this example. The table contains complete target-test means over seeds 0,1,2, not the displayed-window scores. Global MSE is standardized across all four features. Range RMSE is in 100 log(high/low) units. Overall MSE gain over input mean is about 1.60%, and adaptation over the matched strict source-only hybrid adds only about 0.14%. Financial features are not stock-price forecasts or evidence of trading profitability. No new finance training was performed.','0:55','results/strict_refined/finance/metrics_all.csv; results/strict_refined/finance/forecast_examples.npz; models/strict_refined/finance/meta.json; presentation/intensive/evidence.json');
for(const [i,name] of ['hl_range','log_return'].entries()){
 const y=1.63+i*2.32,p=E.finance_plots.features[name],labels=E.finance_plots.days;
 text(s,name==='hl_range'?'Daily high–low range: 100 log(high / low)':'Daily return: 100 log(close / previous close)',.85,y,7.85,.35,18,{bold:true});
 chart(s,[{name:'Actual',labels,values:p.actual},{name:'Source-only',labels,values:p.source},{name:'Adapted',labels,values:p.adapted},{name:'Input mean',labels,values:p.input_mean}],.65,y+.39,8.1,1.88,
  {chartColors:[C.ink,C.saffron,C.teal,C.purple],catAxisLabelFrequency:1,valAxisNumFmt:'0.0',valAxisMajorUnit:name==='hl_range'?.5:.25,valAxisMinVal:name==='hl_range'?1:-.5,valAxisMaxVal:name==='hl_range'?2.5:.5,lineDataSymbol:'circle',lineDataSymbolSize:4});
}
text(s,'Target-test means · 3 seeds',9.0,1.7,3.7,.42,19,{bold:true});
table(s,['Model','MSE','Range RMSE'],[['Input mean',f(E.finance.input_mean.MSE),f(E.finance.input_mean.key_RMSE)],['Source-only',f(E.finance.strict_source.MSE),f(E.finance.strict_source.key_RMSE)],['Adapted',f(E.finance.strict_adapted.MSE),f(E.finance.strict_adapted.key_RMSE)],['Persistence',f(E.finance.persistence.MSE),f(E.finance.persistence.key_RMSE)]],8.95,2.28,3.72,[1.43,1.02,1.27],14,.51);
text(s,`${pct(100*(1-E.finance.strict_adapted.MSE/E.finance.input_mean.MSE))} lower MSE vs input mean`,9.0,5.03,3.6,.55,20,{bold:true,color:C.teal});
text(s,`${pct(100*(1-E.finance.strict_adapted.MSE/E.finance.strict_source.MSE))} from adaptation over source-only`,9.0,5.77,3.6,.58,18,{color:C.slate});
caption(s,'First fixed window, seed 0; trading days 1–5. Mean-like forecasts miss movements. All four features are predicted.');

s=base('Hourly forecasts and peak tradeoffs','Results · matched H=1 comparison',
 'First 96 fixed target origins, seed 0. These are successive next-hour forecasts with refreshed observations, not one 96-hour forecast. Weights stay fixed. Matching source/adapted temperature RMSE gains are 3.84% climate and 27.68% energy. The table is whole-test three-seed means. Full-system reductions over persistence are 48.55% and 31.37%, including source learning and hybrid components. Climate earlier-hybrid first hour is nearly as good (0.654038 RMSE). The first TCN/dynamics energy hybrid has lower overall temperature RMSE 0.896999 but worse event errors. Complete daily-episode peak/trough MAE: climate 0.2810/0.5514, energy 0.8477/0.6650. Energy persistence trough error 0.4761 is better. Final vs first hybrid changes experts/regimes and loss together, so this is not a loss-only ablation.','0:55','results/peak_next_hour/*/rolling_forecasts.npz; results/forecast_horizons/next_hour_headline.csv; results/forecast_horizons/*_next_hour_peak_comparison.csv');
for(const [i,ds] of ['climate','energy'].entries()){const y=1.61+i*2.37,r=E.datasets[ds];text(s,ds==='climate'?'Delhi air temperature (°C)':'ETTh2 oil temperature (°C)',.85,y,7.9,.35,19,{bold:true});chart(s,series(r.H1,'H1'),.65,y+.39,8.1,1.94,{chartColors:[C.ink,C.saffron,C.teal],catAxisLabelFrequency:12});}
text(s,'Temperature RMSE (°C) · 3 seeds',9.0,1.69,3.7,.52,18,{bold:true});
table(s,['Model','Climate','Energy'],[['Source-only',f(E.datasets.climate.hourly.source_key_RMSE,3),f(E.datasets.energy.hourly.source_key_RMSE,3)],['Adapted',f(E.datasets.climate.hourly.adapted_key_RMSE,3),f(E.datasets.energy.hourly.adapted_key_RMSE,3)],['Persistence',f(E.datasets.climate.hourly.persistence_key_RMSE,3),f(E.datasets.energy.hourly.persistence_key_RMSE,3)]],8.95,2.36,3.72,[1.47,1.1,1.15],15,.49);
text(s,'Adaptation-only RMSE reduction',9.0,4.55,3.7,.45,18,{bold:true,color:C.teal});
text(s,'Climate 3.84%    Energy 27.68%',9.0,5.12,3.7,.5,18,{bold:true});
text(s,'Peak / trough MAE (°C)\nClimate: 0.281 / 0.551\nEnergy: 0.848 / 0.665',9.0,5.69,3.7,.81,16,{color:C.slate});
caption(s,'First 96 origins, seed 0. Updated observations each hour. Energy trough MAE still favors persistence (0.476 °C).');

s=base('CSV interface and live demonstration','Submission application',
 'Open Streamlit, select a target CSV and show source/adapted forecasts, actuals and metrics. Switch H=1/H=24 and explain the different tasks. Demonstrate oil temperature and financial feature names. Selected-window scores differ from saved whole-test means. Model weights are fixed. The offline dashboard is a results viewer; the app performs CSV inference.','0:55','app/submission_app.py; src/submission.py; dashboard/IE643_Results_Dashboard.html; local app screenshot, 20 October 2026');
contain(s,A('submission_app.png'),.7,1.58,7.45,4.94);
item(s,'01','Upload a multivariate CSV','Select dataset and source / target domain',8.45,1.85,4.2);item(s,'02','Inspect the forecast','Choose H=24 or rolling H=1',8.45,3.42,4.2);item(s,'03','Export the evidence','Download forecasts, actuals and metrics',8.45,5.0,4.2);

s=base('Conclusions and team contributions','Completed work',
 'Three domain pairs, compact source/adapted models, input-only adaptation, comparative logs/plots and an interface are complete. LiteTCN stays in the hybrid. Hourly accuracy improves, while day-ahead peaks and energy troughs remain open. Roles follow the supplied team statement.','0:25','report/intensive/IE643_Intensive_Assessment_Report.pdf; report/team_details.json');
text(s,'LiteTCN stays. The hybrid improves the hourly task.',.8,1.75,11.8,.95,31,{fontFace:'Cambria',bold:true,color:C.teal});text(s,'Observed-input adaptation improves matched source-only RMSE.\nDay-ahead peaks and energy troughs remain open problems.\nFinance improves only slightly over the strong input-mean baseline.\nThree domain pairs, checkpoints, results and a working CSV interface.',.85,3.05,11.6,1.8,23);line(s,.8,5.35,11.7);
text(s,'Both members',.85,5.72,3.4,.45,20,{bold:true});text(s,'Research and experiments',.85,6.25,3.6,.42,17,{color:C.slate});text(s,'Ayush Patel',4.85,5.72,3.4,.45,20,{bold:true});text(s,'Dashboard and application',4.85,6.25,3.6,.42,17,{color:C.slate});text(s,'Rishik Yadav',8.85,5.72,3.5,.45,20,{bold:true});text(s,'Report and presentation',8.85,6.25,3.6,.42,17,{color:C.slate});

s=base('Plan for the novelty assessment','Next experiments','Plans, not completed results. Lock a fresh chronological period. Hold experts fixed to isolate event loss, then isolate regime conditioning. Investigate energy troughs and uncertainty before distillation.','0:25','report/intensive/report_body.tex; report/IE643_ASSESSMENT_GUIDE.md');
item(s,'01','Fresh chronological evaluation','Separate final confirmation from inspected development tests',.8,1.85);item(s,'02','Controlled event-loss ablation','Hold experts fixed while changing the objective',6.85,1.85);item(s,'03','Regimes and uncertainty','Compare fixed weights and causal regimes; use time blocks',.8,4.15);item(s,'04','Energy dips and compression','Diagnose trough errors and distil useful behavior into a TCN',6.85,4.15);

s=base('References: forecasting and adaptation','Research sources','Papers motivate components and baselines. Custom implementations are not exact reproductions. Full bibliography is also in report/intensive/references.bib.','Reference','report/intensive/references.bib');
const refs=[['Shaojie Bai, J. Zico Kolter and Vladlen Koltun (2018).','An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.'],['Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi and Jaegul Choo (2022).','Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR. openreview.net/forum?id=cGDAkQo1C0p'],['Ailing Zeng, Muxi Chen, Lei Zhang and Qiang Xu (2023).','Are Transformers Effective for Time Series Forecasting? AAAI. arXiv:2205.13504.'],['Baochen Sun and Kate Saenko (2016).','Deep CORAL: Correlation Alignment for Deep Domain Adaptation. ECCV Workshops. arXiv:1607.01719.'],['Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand and Victor Lempitsky (2016).','Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research, 17(59), 1–35.']];
refs.forEach((r,i)=>{let y=1.64+i*.97,authorHeight=i===4?.64:.38;text(s,r[0],.75,y,11.85,authorHeight,15.5,{bold:true,color:C.teal});text(s,r[1],.75,y+authorHeight+.04,11.85,.48,15.5);});

s=base('References: data, code and evidence','Reproducibility sources','Website publication dates are distinct from access dates. n.d. marks a missing stated publication year. The project is the local October 2026 edition; remote synchronization is not implied. Cached files and numerical sources have hashes in the report and evidence export.','Reference','report/intensive/references.bib; src/config.py; presentation/intensive/evidence.json');
const webrefs=[['Open-Meteo contributors. Historical Weather API (n.d.).','open-meteo.com/en/docs/historical-weather-api · Accessed 19 October 2026'],['Haoyi Zhou and contributors. Electricity Transformer Dataset (2021).','github.com/zhouhaoyi/ETDataset · Checked 19 October 2026'],['Yahoo Finance and yfinance contributors. Historical OHLCV data and acquisition code (n.d.).','github.com/ranaroussi/yfinance · Cached project data; code checked 20 October 2026'],['Ayush Patel and Rishik Yadav. IE643 project repository (2026).','github.com/Ayush20990/ie643-project · Local project edition: 20 October 2026']];
webrefs.forEach((r,i)=>{let y=1.65+i*1.05;text(s,r[0],.75,y,11.85,.42,18,{bold:true,color:C.teal});text(s,r[1],.75,y+.49,11.85,.4,17);});
text(s,'Numerical evidence: saved CSVs, forecast arrays and selection manifests in results/.\nChart values and source hashes: presentation/intensive/evidence.json.',.75,6.13,11.85,.67,16,{color:C.slate});

const total=notes.slice(0,18).reduce((t,a)=>{const m=a.match(/\((\d+):(\d+)\)/);return t+Number(m[1])*60+Number(m[2]);},0);
fs.writeFileSync(path.join(out,'IE643_Intensive_Speaker_Notes.md'),'# DeepGen intensive assessment speaker notes\n\n18 content slides + 2 reference slides. Planned talk/demo: '+Math.floor(total/60)+':'+String(total%60).padStart(2,'0')+'. Rehearse within the 12-minute allowance.\n\n'+notes.join('\n'));
fs.writeFileSync(path.join(out,'design_provenance.json'),JSON.stringify({reference:'presentation/IE643_Prep_Presentation.pptx',reference_sha256:require('crypto').createHash('sha256').update(fs.readFileSync(path.join(__dirname,'IE643_Prep_Presentation.pptx'))).digest('hex'),design:'Original contour backgrounds, Cambria headings, Calibri body, teal and saffron. Rebuilt layouts for the intensive content.',slide_size:'16:9, 13.333 × 7.5 inches',native_charts:6,finance_results_slide:14,content_slides:18,reference_slides:2,evidence:'evidence.json',chart_selection:'Fixed first saved origins (H=24 and finance H=5), first 96 origins (H=1), deployment seed 0'},null,2));
async function save(){
 const filename=path.join(out,'IE643_Intensive_Assessment.pptx');
 await pptx.writeFile({fileName:filename});
 // PptxGenJS 4 writes redundant content-type entries for absent slide masters.
 // Remove only declarations whose actual ZIP parts do not exist.
 const JSZip=require('jszip'),zip=await JSZip.loadAsync(fs.readFileSync(filename));
 const types=await zip.file('[Content_Types].xml').async('string');
 zip.file('[Content_Types].xml',types.replace(/<Override\b[^>]*\/>/g,entry=>{
  const part=entry.match(/PartName="\/([^"]+)"/);return part&&!zip.file(part[1])?'':entry;
 }));
 fs.writeFileSync(filename,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
}
save().catch(error=>{console.error(error);process.exitCode=1;});
