"""Novelty finance page embedded in the restored dashboard."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.novelty_finance import ROOT, OUT, DEST, COLS, L, infer, metrics
from src.submission import read_input


def render():
    h=st.radio('Finance forecast target',[1,5],format_func=lambda v:'Next-session range' if v==1 else 'Average range over next 5 sessions',horizontal=True)
    folder=ROOT/('models/novelty_finance' if h==1 else 'models/novelty_finance_mean5')
    out=ROOT/('results/novelty_finance' if h==1 else 'results/novelty_finance_mean5')
    st.subheader('Novelty experiment: trading-range forecasting')
    st.write('Predict 100 × ln(High / Low) for the selected horizon (one daily range or the average of five daily ranges) using the preceding 60 sessions. A larger value means a wider daily price range; this is not a price or buy/sell forecast.')
    if not (folder/'seed_0_both.pt').exists():
        st.info('Training is in progress. Results appear once the model bundle is saved.');return
    with st.sidebar:
        st.header('Finance novelty inputs')
        domain=st.selectbox('Stock basket',['pharma','it'])
        frame=pd.read_csv(ROOT/f'data/processed/finance/{domain}.csv',parse_dates=['date'])
        ticker=st.selectbox('Example company',sorted(frame.series.unique()))
        sample=frame[frame.series==ticker].set_index('date')[COLS].tail(240)
        upload=st.file_uploader('Upload one company CSV',type='csv',key='novelty_upload')
        st.download_button('Download this example',sample.to_csv(index_label='date'),f'{ticker}_novelty.csv','text/csv')
    if upload:
        try:sample=read_input(upload,'finance')
        except Exception as e:st.error(str(e));return
    if len(sample)<L:st.error('At least 60 historical sessions are required.');return
    st.caption('Saved seed 0 for the live demo. Adaptation was fitted on Pharma training inputs only; uploading a file does not retrain the model. Source and adapted models use identical per-window normalization.')
    live=st.checkbox('Predict after the final uploaded observation',value=False)
    if len(sample)<L+h:live=True
    origin=len(sample) if live else st.slider('First predicted row',L,len(sample)-h,len(sample)-h)
    b=json.loads((folder/'seed_0.json').read_text())
    x=sample.iloc[origin-L:origin][COLS].to_numpy(dtype=np.float32)[None]
    pred=infer(x,b,folder)
    labels=['Hybrid source','Hybrid both','Mean 20','Persistence']
    columns=st.columns(4)
    for c,name in zip(columns,labels):c.metric(name,f'{pred[name][0]:.3f}')
    if not live:
        actual=float(sample.iloc[origin:origin+h].hl_range.mean())
        st.write(f'Actual range target: **{actual:.3f}** percentage points. Forecast date: **{sample.index[origin].date()}**.')
    else:st.info('Future actuals are unavailable. The forecast refers to the selected next trading session(s); no calendar date is guessed.')
    fig=go.Figure();hist=sample.iloc[max(0,origin-60):origin]
    fig.add_scatter(x=np.arange(-len(hist),0),y=hist.hl_range,name='Observed range',line=dict(color='#657b88'))
    if not live:fig.add_scatter(x=[0],y=[actual],name='Actual',mode='markers',marker=dict(size=12,color='#102e3a'))
    for name in labels:fig.add_scatter(x=[0],y=[float(pred[name][0])],name=name,mode='markers',marker=dict(size=10))
    fig.update_layout(xaxis_title='History ends at -1; position 0 shows the selected forecast target',yaxis_title='100 ln(High / Low)')
    st.plotly_chart(fig,width='stretch')
    if len(sample)>=L+h and st.button('Evaluate recent observed sessions'):
        ids=np.arange(max(L,len(sample)-h-119),len(sample)-h+1)
        xx=np.stack([sample[COLS].to_numpy(dtype=np.float32)[i-L:i] for i in ids]);pp=infer(xx,b,folder)
        yy=np.array([sample.hl_range.iloc[i:i+h].mean() for i in ids])
        st.dataframe(pd.DataFrame([{'Model':n,**metrics(yy,pp[n])} for n in labels]),hide_index=True)
        result=pd.DataFrame({'date':sample.index[ids],'actual':yy,**{n:pp[n] for n in labels}})
        st.line_chart(result.set_index('date'));st.download_button('Download evaluated forecasts',result.to_csv(index=False),'range_forecasts.csv','text/csv')
    if (out/'metrics.csv').exists():
        st.subheader('Saved benchmark: three seeds')
        m=pd.read_csv(out/'metrics.csv');q=m[m.domain=='target'].groupby('method')[['RMSE','MAE','MSE']].mean()
        st.dataframe(q)
        if (out/'legacy_matched_metrics.csv').exists():
            st.write('Earlier five-step model evaluated on the same target and origins:')
            old=pd.read_csv(out/'legacy_matched_metrics.csv')
            st.dataframe(old.groupby('method')[['RMSE','MAE','MSE']].mean())
            st.caption('For H=5 this averages the earlier model?s five forecasts. Its adapted variant used historical target calibration in the intensive project; the new alignment experiment excludes target forecast labels.')
        st.image(str(out/'comparison.png'))
        st.caption('The earlier finance benchmark predicts four variables over five sessions. These range-only results are a separate task. The five-session option predicts an average, not five individual values. The historical test period was already inspected during the intensive phase.')
    with st.expander('Training and adaptation explained'):
        st.write('The linear branch uses recent range averages and variability. The small causal TCN uses 60 sessions of returns, range, gaps and volume. A blend is selected using IT validation answers only. Source weights are then adjusted with source forecast replay, MMD/CORAL alignment against Pharma input features and a source-weight anchor. Pharma forecast answers are excluded from both fitting and selection. Replay-only, MMD-only and CORAL-only controls are saved alongside the combined method.')
