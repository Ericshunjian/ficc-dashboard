(() => {
  'use strict';
  const colors={sub:'#91a5af',grid:'#20313d',blue:'#4dc4ff',red:'#ff6b7a',orange:'#ffb454'};
  let chart,timer,lastRows=[],lastMult=100;
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
  function bsPut(S,K,T,r,q,sigma){
    if(T<=0){
      const itm=K>S?1:K<S?0:.5;
      return{price:Math.max(K-S,0),delta:K>S?-1:K<S?0:-.5,gamma:0,vega:0,theta:0,d1:NaN,d2:NaN,pitm:itm};
    }
    const root=Math.sqrt(T);
    const d1=(Math.log(S/K)+(r-q+.5*sigma*sigma)*T)/(sigma*root);
    const d2=d1-sigma*root;
    return{
      price:K*Math.exp(-r*T)*normalCdf(-d2)-S*Math.exp(-q*T)*normalCdf(-d1),
      delta:-Math.exp(-q*T)*normalCdf(-d1),
      gamma:Math.exp(-q*T)*normalPdf(d1)/(S*sigma*root),
      vega:S*Math.exp(-q*T)*normalPdf(d1)*root/100,
      theta:(-S*Math.exp(-q*T)*normalPdf(d1)*sigma/(2*root)+q*S*Math.exp(-q*T)*normalCdf(-d1)-r*K*Math.exp(-r*T)*normalCdf(-d2))/365,
      d1,d2,pitm:normalCdf(-d2)
    };
  }
  function probabilityBelow(level,S,T,drift,sigma){
    if(level<=0)return 0;
    if(T<=0)return S<level?1:S>level?0:.5;
    return normalCdf((Math.log(level/S)-(drift-.5*sigma*sigma)*T)/(sigma*Math.sqrt(T)));
  }
  function impliedVol(target,S,K,T,r,q){
    if(!(target>=0&&S>0&&K>0&&T>0))return NaN;
    let lo=.0001,hi=3;
    const min=bsPut(S,K,T,r,q,lo).price;
    const max=bsPut(S,K,T,r,q,hi).price;
    if(target<min-1e-8||target>max+1e-8)return NaN;
    for(let i=0;i<100;i++){
      const mid=(lo+hi)/2;
      if(bsPut(S,K,T,r,q,mid).price<target)lo=mid;else hi=mid;
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
    return{S:val('S'),K:val('K'),premium:val('premium'),mult:val('mult'),days,T:days/365,sigma:val('sigma')/100,r:val('r')/100,q:val('q')/100,kmin:val('kmin'),kmax:val('kmax'),kstep:val('kstep')};
  }
  function valid(p){
    return p.S>0&&p.K>0&&p.premium>=0&&p.mult>0&&p.days>0&&p.sigma>0&&p.kstep>0&&p.kmax>=p.kmin;
  }

  function compute(){
    const p=inputs();
    const status=byId('bsStatus');
    byId('Tdisp').textContent=Number.isFinite(p.days)?`${p.days}个自然日，T=${fmt(p.T,4)}`:'日期无效';
    if(!valid(p)){
      status.textContent='请检查价格、日期与参数';
      status.className='status bad';
      byId('priceKpis').innerHTML='';
      byId('riskKpis').innerHTML='';
      byId('expiryZones').innerHTML='';
      byId('contractFormula').textContent='到期日应晚于估值日，指数、行权价、参考波动率和乘数应大于0。';
      return;
    }

    const fair=bsPut(p.S,p.K,p.T,p.r,p.q,p.sigma);
    const marketIv=impliedVol(p.premium,p.S,p.K,p.T,p.r,p.q);
    const ivOk=Number.isFinite(marketIv);
    const breakEven=Math.max(0,p.K-p.premium);
    const priceDiff=p.premium-fair.price;
    const priceDiffPct=fair.price>1e-10?priceDiff/fair.price:NaN;
    const qDrift=p.r-p.q;
    const exerciseProb=ivOk?probabilityBelow(p.K,p.S,p.T,qDrift,marketIv):NaN;
    const lossProb=ivOk?probabilityBelow(breakEven,p.S,p.T,qDrift,marketIv):NaN;
    const partialProfitProb=ivOk?Math.max(0,exerciseProb-lossProb):NaN;
    const fullPremiumProb=ivOk?Math.max(0,1-exerciseProb):NaN;
    const profitProb=ivOk?Math.max(0,1-lossProb):NaN;
    const maxLoss=Math.max(0,(p.K-p.premium)*p.mult);
    const richer=priceDiff>=0;
    const comparison=richer?'市场价高于你的 BS 参考价，权利金相对更厚':'市场价低于你的 BS 参考价，权利金相对偏薄';

    status.textContent=ivOk?'市场IV已由权利金自动反推':'市场权利金超出BS可反推范围';
    status.className=ivOk?'status':'status bad';
    byId('contractFormula').innerHTML=`${comparison}：市场价 <b>${fmt(p.premium,2)}点</b>，参考价 <b>${fmt(fair.price,2)}点</b>，相差 <b>${signed(priceDiff)}点（${signedPct(priceDiffPct)}）</b>。这只是相对你输入波动率的模型比较，不等于无风险套利。`;

    byId('priceKpis').innerHTML=[
      kpi('市场权利金',`${fmt(p.premium,2)}点`,`${yuan(p.premium*p.mult)} / 张 · 你实际能收到的价格`,'blue'),
      kpi('BS 参考价',`${fmt(fair.price,2)}点`,`${yuan(fair.price*p.mult)} / 张 · 按参考波动率 ${fmt(p.sigma*100,2)}%`,'orange'),
      kpi('市场 − 参考价',`${signed(priceDiff)}点`,`${signedPct(priceDiffPct)} · ${richer?'卖方价格更厚':'卖方价格偏薄'}`,richer?'green':'red'),
      kpi('市场隐含波动率',ivOk?`${fmt(marketIv*100,2)}%`:'无法反推',ivOk?`比参考波动率 ${signed((marketIv-p.sigma)*100,2)} 个波动率点；两者用途不同`:'检查权利金是否低于内在价值或高于理论上限',ivOk?'blue':'red')
    ].join('');

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
      const b=bsPut(p.S,K,p.T,p.r,p.q,p.sigma);
      const be=Math.max(0,K-b.price);
      const rn=probabilityBelow(K,p.S,p.T,qDrift,p.sigma);
      const loss=probabilityBelow(be,p.S,p.T,qDrift,p.sigma);
      const ratio=K/p.S-1;
      const mny=Math.abs(ratio)<=.01?'ATM':K<p.S?'OTM':'ITM';
      rows.push({K,b,be,rn,loss,mny});
    }
    byId('optionRows').innerHTML=rows.map(row=>`<tr data-k="${row.K}" class="${Math.abs(row.K-p.K)<1e-8?'selected':''}"><td>${fmt(row.K,0)}</td><td class="${row.mny==='ITM'?'hit':row.mny==='OTM'?'safe':'orange'}">${row.mny}</td><td><b>${fmt(row.b.price)}</b></td><td>${yuan(row.b.price*p.mult)}</td><td class="neg">${fmt(row.b.delta,3)}</td><td>${fmt(row.b.theta,3)}</td><td>${fmt(row.b.vega,2)}</td><td>${pct(row.rn)}</td><td>${fmt(row.be,2)}</td><td>${pct(row.loss)}</td></tr>`).join('');
    byId('optionRows').querySelectorAll('tr').forEach(tr=>tr.addEventListener('click',()=>{byId('K').value=tr.dataset.k;compute();window.scrollTo({top:0,behavior:'smooth'});}));
    lastRows=rows;
    lastMult=p.mult;
    if(document.querySelector('details.analysis').open)renderChart(rows,p.mult);
  }

  function renderChart(rows,mult){
    if(!rows.length)return;
    const data={labels:rows.map(row=>row.K),datasets:[
      {label:'BS参考合约价（元）',data:rows.map(row=>row.b.price*mult),borderColor:colors.blue,yAxisID:'y',pointRadius:0,borderWidth:2},
      {label:'参考模型：到期实值概率',data:rows.map(row=>row.rn*100),borderColor:colors.orange,yAxisID:'y1',pointRadius:0,borderWidth:2},
      {label:'参考模型：卖方亏损概率',data:rows.map(row=>row.loss*100),borderColor:colors.red,yAxisID:'y1',pointRadius:0,borderWidth:2,borderDash:[5,4]}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:colors.sub}}},scales:{x:{ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'行权价 K',color:colors.sub}},y:{position:'left',ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'合约价（元）',color:colors.sub}},y1:{position:'right',min:0,max:100,ticks:{color:colors.sub,callback:v=>`${v}%`},grid:{drawOnChartArea:false},title:{display:true,text:'到期概率',color:colors.sub}}}};
    if(!chart)chart=new Chart(byId('priceChart'),{type:'line',data,options});else{chart.data=data;chart.update('none');chart.resize();}
  }

  document.querySelectorAll('.hv-btn').forEach(button=>button.addEventListener('click',()=>{byId('sigma').value=button.dataset.hv;compute();}));
  document.querySelectorAll('input').forEach(input=>input.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(compute,100);}));
  document.querySelector('details.analysis').addEventListener('toggle',event=>{if(event.currentTarget.open)requestAnimationFrame(()=>renderChart(lastRows,lastMult));});
  compute();
})();
