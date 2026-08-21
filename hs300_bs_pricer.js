(() => {
  'use strict';
  const colors={sub:'#91a5af',grid:'#20313d',blue:'#4dc4ff',red:'#ff6b7a',orange:'#ffb454',green:'#65d68a'};
  let chart,thetaChart,timer,lastRows=[],lastMult=100;
  const byId=id=>document.getElementById(id);
  const raw=id=>byId(id).value;
  const val=id=>Number(raw(id));
  const normalPdf=x=>Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI);
  const fmt=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
  const pct=(x,d=1)=>Number.isFinite(x)?`${(100*x).toFixed(d)}%`:'—';
  const signedPct=(x,d=1)=>Number.isFinite(x)?`${x>=0?'+':''}${(100*x).toFixed(d)}%`:'—';
  const signed=(x,d=2)=>Number.isFinite(x)?`${x>=0?'+':''}${x.toFixed(d)}`:'—';
  const yuan=x=>Number.isFinite(x)?`¥${Math.round(x).toLocaleString()}`:'—';

  function normalCdf(x){
    const t=1/(1+.2316419*Math.abs(x));
    const d=.3989422804014327*Math.exp(-.5*x*x);
    const p=d*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));
    return x>=0?1-p:p;
  }
  function dayCount(a,b){
    if(!a||!b)return NaN;
    return Math.round((Date.parse(`${b}T00:00:00Z`)-Date.parse(`${a}T00:00:00Z`))/86400000);
  }
  function blackPut(F,K,T,r,sigma){
    if(T<=0){
      const itm=K>F?1:K<F?0:.5;
      return{price:Math.max(K-F,0),delta:K>F?-1:K<F?0:-.5,vega:0,theta:0,d1:NaN,d2:NaN,pitm:itm};
    }
    const root=Math.sqrt(T);
    const d1=(Math.log(F/K)+.5*sigma*sigma*T)/(sigma*root);
    const d2=d1-sigma*root;
    const discount=Math.exp(-r*T);
    const price=discount*(K*normalCdf(-d2)-F*normalCdf(-d1));
    return{
      price,
      delta:-discount*normalCdf(-d1),
      vega:discount*F*normalPdf(d1)*root/100,
      theta:(r*price-discount*F*normalPdf(d1)*sigma/(2*root))/365,
      d1,d2,pitm:normalCdf(-d2)
    };
  }
  function spotPut(S,K,T,r,q,sigma){
    if(T<=0){
      const itm=K>S?1:K<S?0:.5;
      return{price:Math.max(K-S,0),delta:K>S?-1:K<S?0:-.5,vega:0,theta:0,d1:NaN,d2:NaN,pitm:itm};
    }
    const root=Math.sqrt(T);
    const d1=(Math.log(S/K)+(r-q+.5*sigma*sigma)*T)/(sigma*root);
    const d2=d1-sigma*root;
    const discountR=Math.exp(-r*T);
    const discountQ=Math.exp(-q*T);
    const price=K*discountR*normalCdf(-d2)-S*discountQ*normalCdf(-d1);
    return{
      price,
      delta:-discountQ*normalCdf(-d1),
      vega:S*discountQ*normalPdf(d1)*root/100,
      theta:(-S*discountQ*normalPdf(d1)*sigma/(2*root)+r*K*discountR*normalCdf(-d2)-q*S*discountQ*normalCdf(-d1))/365,
      d1,d2,pitm:normalCdf(-d2)
    };
  }
  function probabilityBelow(level,F,T,sigma){
    if(level<=0)return 0;
    if(T<=0)return F<level?1:F>level?0:.5;
    return normalCdf((Math.log(level/F)+.5*sigma*sigma*T)/(sigma*Math.sqrt(T)));
  }
  function impliedVol(target,F,K,T,r){
    if(!(target>=0&&F>0&&K>0&&T>0))return NaN;
    let lo=.0001,hi=3;
    const min=blackPut(F,K,T,r,lo).price;
    const max=blackPut(F,K,T,r,hi).price;
    if(target<min-1e-8||target>max+1e-8)return NaN;
    for(let i=0;i<100;i++){
      const mid=(lo+hi)/2;
      if(blackPut(F,K,T,r,mid).price<target)lo=mid;else hi=mid;
    }
    return (lo+hi)/2;
  }
  function impliedCarry(target,S,K,T,r,sigma){
    if(!(target>=0&&S>0&&K>0&&T>0&&sigma>0))return NaN;
    let lo=-1,hi=3;
    if(target<spotPut(S,K,T,r,lo,sigma).price-1e-8||target>spotPut(S,K,T,r,hi,sigma).price+1e-8)return NaN;
    for(let i=0;i<100;i++){
      const mid=(lo+hi)/2;
      if(spotPut(S,K,T,r,mid,sigma).price<target)lo=mid;else hi=mid;
    }
    return (lo+hi)/2;
  }
  function kpi(label,value,detail,cls='blue'){
    return `<div class="kpi"><div class="l">${label}</div><div class="v ${cls}">${value}</div><div class="d">${detail}</div></div>`;
  }
  function zone(cls,range,meaning,probability){
    return `<div class="expiry-zone ${cls}"><div><div class="range">${range}</div><div class="meaning">${meaning}</div></div><div class="prob ${cls==='loss'?'red':cls==='partial'?'orange':'green'}">${pct(probability)}</div></div>`;
  }
  function inputs(){
    const days=dayCount(raw('today'),raw('expiry'));
    const enteredForward=raw('F').trim()===''?NaN:val('F');
    const enteredMarketIv=raw('marketIv').trim()===''?NaN:val('marketIv')/100;
    return{S:val('S'),enteredForward,enteredMarketIv,K:val('K'),premium:val('premium'),mult:val('mult'),days,T:days/365,sigma:val('sigma')/100,r:val('r')/100,q:val('q')/100,kmin:val('kmin'),kmax:val('kmax'),kstep:val('kstep')};
  }
  function valid(p){
    return p.S>0&&(!Number.isFinite(p.enteredForward)||p.enteredForward>0)&&(!Number.isFinite(p.enteredMarketIv)||p.enteredMarketIv>0)&&p.K>0&&p.premium>=0&&p.mult>0&&p.days>0&&p.sigma>0&&p.kstep>0&&p.kmax>=p.kmin;
  }

  function compute(){
    const p=inputs();
    const status=byId('bsStatus');
    byId('Tdisp').textContent=Number.isFinite(p.days)?`${p.days}个自然日，T=${fmt(p.T,4)}`:'日期无效';
    if(!valid(p)){
      status.textContent='请检查价格、日期与参数';
      status.className='status bad';
      byId('priceKpis').innerHTML='';
      byId('thetaKpis').innerHTML='';
      byId('riskKpis').innerHTML='';
      byId('expiryZones').innerHTML='';
      if(thetaChart){thetaChart.destroy();thetaChart=null;}
      byId('contractFormula').textContent='到期日应晚于估值日，指数、行权价、参考波动率和乘数应大于0。';
      return;
    }

    const usesMarketForward=Number.isFinite(p.enteredForward);
    const F=usesMarketForward?p.enteredForward:p.S*Math.exp((p.r-p.q)*p.T);
    const forwardSource=usesMarketForward?'同到期IF市场远期':'现货、利率和股息率估算远期';
    const fair=blackPut(F,p.K,p.T,p.r,p.sigma);
    const forwardIv=impliedVol(p.premium,F,p.K,p.T,p.r);
    const ivOk=Number.isFinite(forwardIv);
    const breakEven=Math.max(0,p.K-p.premium);
    const priceDiff=p.premium-fair.price;
    const priceDiffPct=fair.price>1e-10?priceDiff/fair.price:NaN;
    const exerciseProb=ivOk?probabilityBelow(p.K,F,p.T,forwardIv):NaN;
    const lossProb=ivOk?probabilityBelow(breakEven,F,p.T,forwardIv):NaN;
    const partialProfitProb=ivOk?Math.max(0,exerciseProb-lossProb):NaN;
    const fullPremiumProb=ivOk?Math.max(0,1-exerciseProb):NaN;
    const profitProb=ivOk?Math.max(0,1-lossProb):NaN;
    const maxLoss=Math.max(0,(p.K-p.premium)*p.mult);
    const richer=priceDiff>=0;
    const comparison=richer?'市场价高于你的模型参考价，权利金相对更厚':'市场价低于你的模型参考价，权利金相对偏薄';

    if(!ivOk){status.textContent='权利金超出当前远期口径的可反推范围';status.className='status bad';}
    else if(usesMarketForward){status.textContent=`使用IF市场远期 F=${fmt(F,2)}`;status.className='status';}
    else{status.textContent=`未填IF，使用估算远期 F=${fmt(F,2)}`;status.className='status warn';}
    byId('contractFormula').innerHTML=`当前采用<b>${forwardSource} F=${fmt(F,2)}</b>。${comparison}：市场价 <b>${fmt(p.premium,2)}点</b>，参考价 <b>${fmt(fair.price,2)}点</b>，相差 <b>${signed(priceDiff)}点（${signedPct(priceDiffPct)}）</b>。`;

    byId('priceKpis').innerHTML=[
      kpi('市场权利金',`${fmt(p.premium,2)}点`,`${yuan(p.premium*p.mult)} / 张 · 你实际能收到的价格`,'blue'),
      kpi('模型参考价',`${fmt(fair.price,2)}点`,`${yuan(fair.price*p.mult)} / 张 · ${forwardSource}`,'orange'),
      kpi('市场 − 参考价',`${signed(priceDiff)}点`,`${signedPct(priceDiffPct)} · ${richer?'卖方价格更厚':'卖方价格偏薄'}`,richer?'green':'red'),
      kpi('IF口径反推IV',ivOk?`${fmt(forwardIv*100,2)}%`:'无法反推',ivOk?`比参考波动率 ${signed((forwardIv-p.sigma)*100,2)} 个波动率点`:'检查权利金、远期和日期是否同一时点',ivOk?'blue':'red')
    ].join('');

    const manualIvOk=Number.isFinite(p.enteredMarketIv);
    const thetaSigma=manualIvOk?p.enteredMarketIv:ivOk?forwardIv:p.sigma;
    const marketImpliedQ=manualIvOk?impliedCarry(p.premium,p.S,p.K,p.T,p.r,thetaSigma):NaN;
    const forwardImpliedQ=p.r-Math.log(F/p.S)/p.T;
    const thetaQ=Number.isFinite(marketImpliedQ)?marketImpliedQ:forwardImpliedQ;
    const thetaNow=spotPut(p.S,p.K,p.T,p.r,thetaQ,thetaSigma);
    const thetaNext=spotPut(p.S,p.K,Math.max(0,(p.days-1)/365),p.r,thetaQ,thetaSigma);
    const fixedForwardTheta=blackPut(F,p.K,p.T,p.r,thetaSigma).theta;
    const exactDecay=thetaNow.price-thetaNext.price;
    const sellerDaily=-thetaNow.theta*p.mult;
    const thetaBasis=Number.isFinite(marketImpliedQ)?`成交价+成交IV反推分红/持有收益q ${(thetaQ*100).toFixed(2)}%`:`由IF反推分红/持有收益q ${(thetaQ*100).toFixed(2)}%`;
    byId('thetaKpis').innerHTML=[
      kpi('买方 Theta（Wind近似）',`${fmt(thetaNow.theta,2)}点/日`,`Wind一位小数约 ${fmt(thetaNow.theta,1)}；${thetaBasis}`,'red'),
      kpi('卖方理论日收入',yuan(sellerDaily),`每手；明日理论价 ${fmt(thetaNext.price,2)}点，有限差分 ${fmt(exactDecay,2)}点`,'green'),
      kpi('固定IF Theta',`${fmt(fixedForwardTheta,2)}点/日`,`原网页Black-76口径，供对照`,'blue'),
      kpi('Theta 采用的 IV',`${fmt(thetaSigma*100,2)}%`,manualIvOk?'采用手工输入的市场成交IV':ivOk?'由当前市场权利金反推':'暂用参考波动率','orange')
    ].join('');
    renderThetaChart(p,thetaQ,thetaSigma);

    byId('expiryZones').innerHTML=[
      zone('loss',`Sₜ < ${fmt(breakEven,2)}`,'卖方净亏损',lossProb),
      zone('partial',`${fmt(breakEven,2)} ≤ Sₜ < ${fmt(p.K,2)}`,'会履约，但权利金仍覆盖亏损',partialProfitProb),
      zone('full',`Sₜ ≥ ${fmt(p.K,2)}`,'Put作废，权利金全赚',fullPremiumProb)
    ].join('');
    byId('riskKpis').innerHTML=[
      kpi('盈亏平衡点 K − C',fmt(breakEven,2),`比卖出时指数低 ${pct(1-breakEven/p.S,2)}`,'green'),
      kpi('到期履约概率',pct(exerciseProb),`Sₜ < K；Put到期实值并现金结算`,'orange'),
      kpi('卖方到期盈利概率',pct(profitProb),`Sₜ > K−C；含“履约但仍盈利”`,'green'),
      kpi('卖方到期亏损概率',pct(lossProb),`Sₜ < K−C；不含手续费和资金成本`,'red'),
      kpi('理论最大亏损',yuan(maxLoss),`每张；极端假设到期指数为0`,'red')
    ].join('');

    const rows=[];
    for(let K=p.kmin,count=0;K<=p.kmax+1e-9&&count<201;K+=p.kstep,count++){
      const b=blackPut(F,K,p.T,p.r,p.sigma);
      const be=Math.max(0,K-b.price);
      const rn=probabilityBelow(K,F,p.T,p.sigma);
      const loss=probabilityBelow(be,F,p.T,p.sigma);
      const ratio=K/F-1;
      const mny=Math.abs(ratio)<=.01?'ATM':K<F?'OTM':'ITM';
      rows.push({K,b,be,rn,loss,mny});
    }
    byId('optionRows').innerHTML=rows.map(row=>`<tr data-k="${row.K}" class="${Math.abs(row.K-p.K)<1e-8?'selected':''}"><td>${fmt(row.K,0)}</td><td class="${row.mny==='ITM'?'hit':row.mny==='OTM'?'safe':'orange'}">${row.mny}</td><td><b>${fmt(row.b.price)}</b></td><td>${yuan(row.b.price*p.mult)}</td><td class="neg">${fmt(row.b.delta,3)}</td><td>${fmt(row.b.vega,2)}</td><td>${pct(row.rn)}</td><td>${fmt(row.be,2)}</td><td>${pct(row.loss)}</td></tr>`).join('');
    byId('optionRows').querySelectorAll('tr').forEach(tr=>tr.addEventListener('click',()=>{byId('K').value=tr.dataset.k;compute();window.scrollTo({top:0,behavior:'smooth'});}));
    lastRows=rows;
    lastMult=p.mult;
    if(document.querySelector('details.analysis').open)renderChart(rows,p.mult);
  }

  function renderChart(rows,mult){
    if(!rows.length)return;
    const data={labels:rows.map(row=>row.K),datasets:[
      {label:'模型参考合约价（元）',data:rows.map(row=>row.b.price*mult),borderColor:colors.blue,yAxisID:'y',pointRadius:0,borderWidth:2},
      {label:'参考模型：到期实值概率',data:rows.map(row=>row.rn*100),borderColor:colors.orange,yAxisID:'y1',pointRadius:0,borderWidth:2},
      {label:'参考模型：卖方亏损概率',data:rows.map(row=>row.loss*100),borderColor:colors.red,yAxisID:'y1',pointRadius:0,borderWidth:2,borderDash:[5,4]}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:colors.sub}}},scales:{x:{ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'行权价 K',color:colors.sub}},y:{position:'left',ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'合约价（元）',color:colors.sub}},y1:{position:'right',min:0,max:100,ticks:{color:colors.sub,callback:v=>`${v}%`},grid:{drawOnChartArea:false},title:{display:true,text:'到期概率',color:colors.sub}}}};
    if(!chart)chart=new Chart(byId('priceChart'),{type:'line',data,options});else{chart.data=data;chart.update('none');chart.resize();}
  }

  function renderThetaChart(p,q,sigma){
    const labels=[],income=[],prices=[];
    for(let elapsed=0;elapsed<p.days;elapsed++){
      const remaining=p.days-elapsed;
      const b=spotPut(p.S,p.K,remaining/365,p.r,q,sigma);
      labels.push(`${remaining}天`);
      income.push(-b.theta*p.mult);
      prices.push(b.price);
    }
    const data={labels,datasets:[
      {label:'卖方理论日收入（元/手）',data:income,borderColor:colors.green,backgroundColor:'rgba(101,214,138,.10)',fill:true,yAxisID:'y',pointRadius:0,borderWidth:2},
      {label:'期权理论价（点）',data:prices,borderColor:colors.orange,yAxisID:'y1',pointRadius:0,borderWidth:2,borderDash:[5,4]}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:colors.sub}},tooltip:{callbacks:{title:items=>`剩余 ${items[0].label}`}}},scales:{x:{ticks:{color:colors.sub,maxTicksLimit:10},grid:{color:colors.grid},title:{display:true,text:'时间向右推进：当前 → 到期前1天',color:colors.sub}},y:{position:'left',ticks:{color:colors.sub,callback:v=>`¥${v}`},grid:{color:colors.grid},title:{display:true,text:'卖方日Theta收入（元/手）',color:colors.sub}},y1:{position:'right',ticks:{color:colors.sub},grid:{drawOnChartArea:false},title:{display:true,text:'期权理论价（点）',color:colors.sub}}}};
    if(!thetaChart)thetaChart=new Chart(byId('thetaChart'),{type:'line',data,options});else{thetaChart.data=data;thetaChart.update('none');thetaChart.resize();}
  }

  function localIsoDate(){
    const now=new Date();
    return new Date(now.getTime()-now.getTimezoneOffset()*60000).toISOString().slice(0,10);
  }
  byId('today').value=localIsoDate();
  document.querySelectorAll('.hv-btn').forEach(button=>button.addEventListener('click',()=>{byId('sigma').value=button.dataset.hv;compute();}));
  document.querySelectorAll('input').forEach(input=>input.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(compute,100);}));
  document.querySelector('details.analysis').addEventListener('toggle',event=>{if(event.currentTarget.open)requestAnimationFrame(()=>renderChart(lastRows,lastMult));});
  compute();
})();
