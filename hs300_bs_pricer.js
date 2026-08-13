(() => {
  'use strict';
  const colors={sub:'#91a5af',grid:'#20313d',blue:'#4dc4ff',red:'#ff6b7a',orange:'#ffb454',green:'#65d68a'};
  let chart, timer;
  const byId=id=>document.getElementById(id);
  const raw=id=>byId(id).value;
  const val=id=>Number(raw(id));
  const normalPdf=x=>Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI);
  const fmt=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
  const pct=(x,d=1)=>Number.isFinite(x)?`${(100*x).toFixed(d)}%`:'—';
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
    if(T<=0){const itm=K>S?1:K<S?0:.5;return{price:Math.max(K-S,0),delta:K>S?-1:K<S?0:-.5,gamma:0,vega:0,theta:0,d1:NaN,d2:NaN,pitm:itm};}
    const root=Math.sqrt(T),d1=(Math.log(S/K)+(r-q+.5*sigma*sigma)*T)/(sigma*root),d2=d1-sigma*root;
    return{
      price:K*Math.exp(-r*T)*normalCdf(-d2)-S*Math.exp(-q*T)*normalCdf(-d1),
      delta:-Math.exp(-q*T)*normalCdf(-d1),
      gamma:Math.exp(-q*T)*normalPdf(d1)/(S*sigma*root),
      vega:S*Math.exp(-q*T)*normalPdf(d1)*root/100,
      theta:(-S*Math.exp(-q*T)*normalPdf(d1)*sigma/(2*root)+r*K*Math.exp(-r*T)*normalCdf(-d2)-q*S*Math.exp(-q*T)*normalCdf(-d1))/365,
      d1,d2,pitm:normalCdf(-d2)
    };
  }
  function probabilityBelow(level,S,T,drift,sigma){
    if(level<=0)return 0;
    if(T<=0)return S<level?1:S>level?0:.5;
    return normalCdf((Math.log(level/S)-(drift-.5*sigma*sigma)*T)/(sigma*Math.sqrt(T)));
  }
  function expectedPutPayoff(K,S,T,mu,sigma){
    if(T<=0)return Math.max(K-S,0);
    const root=Math.sqrt(T),d1=(Math.log(S/K)+(mu+.5*sigma*sigma)*T)/(sigma*root),d2=d1-sigma*root;
    return K*normalCdf(-d2)-S*Math.exp(mu*T)*normalCdf(-d1);
  }
  function impliedVol(target,S,K,T,r,q){
    if(!(target>=0&&S>0&&K>0&&T>0))return NaN;
    let lo=.0001,hi=3;
    const min=bsPut(S,K,T,r,q,lo).price,max=bsPut(S,K,T,r,q,hi).price;
    if(target<min-1e-8||target>max+1e-8)return NaN;
    for(let i=0;i<100;i++){const mid=(lo+hi)/2;if(bsPut(S,K,T,r,q,mid).price<target)lo=mid;else hi=mid;}
    return (lo+hi)/2;
  }
  function kpi(label,value,detail,cls='blue'){return `<div class="kpi"><div class="l">${label}</div><div class="v ${cls}">${value}</div><div class="d">${detail}</div></div>`;}

  function inputs(){
    const days=dayCount(raw('today'),raw('expiry'));
    return{S:val('S'),K:val('K'),premium:raw('premium').trim()===''?NaN:val('premium'),mult:val('mult'),days,T:days/365,sigma:val('sigma')/100,r:val('r')/100,q:val('q')/100,mu:val('mu')/100,kmin:val('kmin'),kmax:val('kmax'),kstep:val('kstep')};
  }
  function valid(p){return p.S>0&&p.K>0&&p.mult>0&&p.days>=0&&p.sigma>0&&p.kstep>0&&p.kmax>=p.kmin;}

  function compute(){
    const p=inputs(),status=byId('bsStatus');
    byId('Tdisp').textContent=Number.isFinite(p.days)?`${p.days}个自然日，T=${fmt(p.T,4)}`:'日期无效';
    if(!valid(p)){status.textContent='请检查日期与参数';status.className='status bad';byId('contractKpis').innerHTML='';return;}
    status.textContent='BS欧式Put · 连续复利';status.className='status';
    const selected=bsPut(p.S,p.K,p.T,p.r,p.q,p.sigma);
    const premium=Number.isFinite(p.premium)?p.premium:selected.price;
    const breakEven=Math.max(0,p.K-premium);
    const qDrift=p.r-p.q;
    const rnItm=probabilityBelow(p.K,p.S,p.T,qDrift,p.sigma);
    const realItm=probabilityBelow(p.K,p.S,p.T,p.mu,p.sigma);
    const rnLoss=probabilityBelow(breakEven,p.S,p.T,qDrift,p.sigma);
    const realLoss=probabilityBelow(breakEven,p.S,p.T,p.mu,p.sigma);
    const physicalPnL=(premium-expectedPutPayoff(p.K,p.S,p.T,p.mu,p.sigma))*p.mult;
    const source=Number.isFinite(p.premium)?'实际输入权利金':'未输入市场价，暂用理论价';
    byId('contractFormula').innerHTML=`选定 IO Put：S=${fmt(p.S)}，K=${fmt(p.K,0)}，${p.days}天，σ=${fmt(p.sigma*100,2)}%。<b>盈亏平衡点 = K − 权利金 = ${fmt(breakEven,2)}</b>（较现货 ${pct(breakEven/p.S-1,2)}）。${source}。`;
    byId('contractKpis').innerHTML=[
      kpi('BS理论权利金',`${fmt(selected.price,2)}点`,`${yuan(selected.price*p.mult)} / 张 · 实际采用 ${fmt(premium,2)}点`,'blue'),
      kpi('盈亏平衡点',fmt(breakEven,2),`到期 Sₜ < ${fmt(breakEven,2)} 才产生净亏损`,'green'),
      kpi('风险中性：到期实值',pct(rnItm),`N(−d₂)，是定价测度，不是预测`,'orange'),
      kpi('现实假设：到期实值',pct(realItm),`基于价格漂移 μ=${fmt(p.mu*100,1)}%`,'blue'),
      kpi('风险中性：卖方到期亏损',pct(rnLoss),`阈值使用 K−权利金，不是K`,'red'),
      kpi('现实假设：卖方到期亏损',pct(realLoss),`到期盈利概率约 ${pct(1-realLoss)}`,'red'),
      kpi('Put Delta',fmt(selected.delta,3),`Gamma ${fmt(selected.gamma,5)} · Vega ${fmt(selected.vega,2)}点/vol`,'orange'),
      kpi('现实假设：模型期望到期盈亏',yuan(physicalPnL),`每张；依赖 μ 与对数正态假设` ,physicalPnL>=0?'green':'red')
    ].join('');

    const rows=[];
    const maxRows=201;
    for(let K=p.kmin,count=0;K<=p.kmax+1e-9&&count<maxRows;K+=p.kstep,count++){
      const b=bsPut(p.S,K,p.T,p.r,p.q,p.sigma),be=Math.max(0,K-b.price);
      const rn=probabilityBelow(K,p.S,p.T,qDrift,p.sigma),real=probabilityBelow(K,p.S,p.T,p.mu,p.sigma),loss=probabilityBelow(be,p.S,p.T,qDrift,p.sigma);
      const ratio=K/p.S-1,mny=Math.abs(ratio)<=.01?'ATM':K<p.S?'OTM':'ITM';
      rows.push({K,b,be,rn,real,loss,mny});
    }
    byId('optionRows').innerHTML=rows.map(r=>`<tr data-k="${r.K}" class="${Math.abs(r.K-p.K)<1e-8?'selected':''}"><td>${fmt(r.K,0)}</td><td class="${r.mny==='ITM'?'hit':r.mny==='OTM'?'safe':'orange'}">${r.mny}</td><td><b>${fmt(r.b.price)}</b></td><td>${yuan(r.b.price*p.mult)}</td><td class="neg">${fmt(r.b.delta,3)}</td><td>${fmt(r.b.theta,3)}</td><td>${fmt(r.b.vega,2)}</td><td>${pct(r.rn)}</td><td>${pct(r.real)}</td><td>${fmt(r.be,2)}</td><td>${pct(r.loss)}</td></tr>`).join('');
    byId('optionRows').querySelectorAll('tr').forEach(tr=>tr.addEventListener('click',()=>{byId('K').value=tr.dataset.k;compute();}));
    renderChart(rows,p.mult);
  }

  function renderChart(rows,mult){
    const data={labels:rows.map(r=>r.K),datasets:[
      {label:'理论合约价（元）',data:rows.map(r=>r.b.price*mult),borderColor:colors.blue,yAxisID:'y',pointRadius:0,borderWidth:2},
      {label:'风险中性到期实值概率',data:rows.map(r=>r.rn*100),borderColor:colors.orange,yAxisID:'y1',pointRadius:0,borderWidth:2},
      {label:'风险中性卖方亏损概率',data:rows.map(r=>r.loss*100),borderColor:colors.red,yAxisID:'y1',pointRadius:0,borderWidth:2,borderDash:[5,4]}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:colors.sub}}},scales:{x:{ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'行权价 K',color:colors.sub}},y:{position:'left',ticks:{color:colors.sub},grid:{color:colors.grid},title:{display:true,text:'合约价（元）',color:colors.sub}},y1:{position:'right',min:0,max:100,ticks:{color:colors.sub,callback:v=>`${v}%`},grid:{drawOnChartArea:false},title:{display:true,text:'到期概率',color:colors.sub}}}};
    if(!chart)chart=new Chart(byId('priceChart'),{type:'line',data,options});else{chart.data=data;chart.update('none');}
  }

  byId('modelPremium').addEventListener('click',()=>{byId('premium').value='';compute();});
  byId('solveIv').addEventListener('click',()=>{
    const p=inputs();
    if(!Number.isFinite(p.premium)){byId('bsStatus').textContent='先输入实际权利金';byId('bsStatus').className='status warn';return;}
    const iv=impliedVol(p.premium,p.S,p.K,p.T,p.r,p.q);
    if(!Number.isFinite(iv)){byId('bsStatus').textContent='权利金超出模型可反推范围';byId('bsStatus').className='status bad';return;}
    byId('sigma').value=(iv*100).toFixed(3);compute();byId('bsStatus').textContent=`已反推 IV ${(iv*100).toFixed(3)}%`;
  });
  document.querySelectorAll('.hv-btn').forEach(b=>b.addEventListener('click',()=>{byId('sigma').value=b.dataset.hv;compute();}));
  document.querySelectorAll('input').forEach(el=>el.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(compute,120);}));
  compute();
})();
