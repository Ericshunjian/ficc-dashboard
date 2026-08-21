(() => {
  'use strict';

  const byId=id=>document.getElementById(id);
  const fmt=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
  const pctText=x=>Number.isFinite(x)?`P${fmt(x,1)}`:'—';
  const colors={blue:'#4dc4ff',green:'#65d68a',orange:'#ffb454',red:'#ff6b7a',cyan:'#50e3c2',sub:'#91a5af',grid:'#20313d'};
  let payload,distributionChart,seriesChart;

  const markerPlugin={
    id:'hvMarkers',
    afterDatasetsDraw(chart,args,options){
      const items=(options&&options.items)||[];
      const {ctx,chartArea,scales}=chart;
      if(!chartArea||!scales.x)return;
      ctx.save();
      items.forEach((item,index)=>{
        if(!Number.isFinite(item.value)||item.value<scales.x.min||item.value>scales.x.max)return;
        const x=scales.x.getPixelForValue(item.value);
        ctx.strokeStyle=item.color;ctx.lineWidth=1.5;ctx.setLineDash(item.dash||[]);
        ctx.beginPath();ctx.moveTo(x,chartArea.top);ctx.lineTo(x,chartArea.bottom);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle=item.color;ctx.font='11px Inter, "Microsoft YaHei", sans-serif';
        ctx.textAlign=x>chartArea.right-90?'right':x<chartArea.left+90?'left':'center';
        const labelX=x>chartArea.right-90?x-3:x<chartArea.left+90?x+3:x;
        ctx.fillText(`${item.label} ${fmt(item.value,2)}%`,labelX,chartArea.top+12+index*13);
      });
      ctx.restore();
    }
  };
  Chart.register(markerPlugin);

  function mean(values){return values.reduce((sum,value)=>sum+value,0)/values.length;}
  function std(values,avg=mean(values)){return values.length<2?0:Math.sqrt(values.reduce((sum,value)=>sum+(value-avg)**2,0)/(values.length-1));}
  function quantile(sorted,q){
    if(!sorted.length)return NaN;
    const position=(sorted.length-1)*q,lower=Math.floor(position),upper=Math.ceil(position);
    return lower===upper?sorted[lower]:sorted[lower]+(sorted[upper]-sorted[lower])*(position-lower);
  }
  function percentileRank(sorted,value){
    let below=0,equal=0;
    for(const item of sorted){if(item<value)below++;else if(Math.abs(item-value)<1e-9)equal++;}
    return (below+.5*equal)/sorted.length*100;
  }
  function normalPdf(x,avg,s){if(!(s>0))return 0;const z=(x-avg)/s;return Math.exp(-.5*z*z)/(s*Math.sqrt(2*Math.PI));}
  function kde(x,values,bandwidth){const scale=bandwidth*Math.sqrt(2*Math.PI);return values.reduce((sum,value)=>sum+Math.exp(-.5*((x-value)/bandwidth)**2),0)/(values.length*scale);}
  function kpi(label,value,detail,cls='blue'){return `<div class="kpi"><div class="l">${label}</div><div class="v ${cls}">${value}</div><div class="d">${detail}</div></div>`;}

  function histogram(values){
    const sorted=[...values].sort((a,b)=>a-b),n=sorted.length,min=sorted[0],max=sorted[n-1],avg=mean(sorted),s=std(sorted,avg);
    const iqr=quantile(sorted,.75)-quantile(sorted,.25);
    let width=2*iqr/Math.cbrt(n);if(!(width>0))width=3.49*s/Math.cbrt(n);
    let bins=Math.round((max-min)/(width||1));bins=Math.max(14,Math.min(36,bins||14));
    width=(max-min||1)/bins;
    const lower=Math.max(0,min-width*.55),upper=max+width*.55; width=(upper-lower)/bins;
    const counts=Array(bins).fill(0);
    sorted.forEach(value=>counts[Math.min(bins-1,Math.max(0,Math.floor((value-lower)/width)))]++);
    const bars=counts.map((count,index)=>({x:lower+(index+.5)*width,y:count/(n*width),count,share:count/n*100,left:lower+index*width,right:lower+(index+1)*width}));
    const bandwidth=Math.max(width*.7,.9*Math.min(s,iqr/1.34||s)*Math.pow(n,-.2));
    const grid=Array.from({length:141},(_,index)=>lower+(upper-lower)*index/140);
    return{sorted,avg,s,bars,grid,kde:grid.map(x=>({x,y:kde(x,sorted,bandwidth)})),normal:grid.map(x=>({x,y:normalPdf(x,avg,s)})),lower,upper};
  }
  function readNumber(id){const text=byId(id).value.trim();return text===''?NaN:Number(text);}
  function filteredRows(window){
    const start=byId('startDate').value,end=byId('endDate').value;
    if(!start||!end||start>end)return[];
    const values=payload[`hv${window}`];
    return payload.dates.map((date,index)=>({date,value:values[index],close:payload.close[index]})).filter(row=>row.date>=start&&row.date<=end&&Number.isFinite(row.value));
  }
  function markers(stats,current){
    const items=[
      {label:'区间末HV',value:current,color:colors.red},
      {label:'中位数',value:quantile(stats.sorted,.5),color:colors.cyan,dash:[4,3]},
      {label:'P90',value:quantile(stats.sorted,.9),color:colors.orange,dash:[5,4]}
    ];
    const atm=readNumber('atmIv'),put=readNumber('putIv');
    if(atm>0)items.push({label:'ATM IV',value:atm,color:colors.blue,dash:[3,3]});
    if(put>0)items.push({label:'Put IV',value:put,color:colors.green,dash:[7,3]});
    return items;
  }
  function renderDistribution(window,rows,stats){
    const current=rows[rows.length-1].value;
    const markerItems=markers(stats,current);
    const markerValues=markerItems.map(item=>item.value).filter(Number.isFinite);
    const visibleMin=Math.min(stats.lower,...markerValues),visibleMax=Math.max(stats.upper,...markerValues);
    const padding=Math.max((visibleMax-visibleMin)*.035,.35);
    const data={datasets:[
      {type:'bar',label:'历史密度',data:stats.bars,parsing:false,backgroundColor:'rgba(77,196,255,.28)',borderColor:colors.blue,borderWidth:1,barPercentage:1,categoryPercentage:1},
      {type:'line',label:'KDE实际分布',data:stats.kde,parsing:false,borderColor:colors.green,pointRadius:0,borderWidth:2.4,tension:.22},
      {type:'line',label:'正态分布对照',data:stats.normal,parsing:false,borderColor:colors.orange,pointRadius:0,borderWidth:1.7,borderDash:[6,5],tension:.2}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'nearest',intersect:false},plugins:{legend:{display:false},hvMarkers:{items:markerItems},tooltip:{callbacks:{label:context=>{
      if(context.dataset.type==='bar'){const raw=context.raw;return `${fmt(raw.left,1)}–${fmt(raw.right,1)}%：${raw.count}次（${fmt(raw.share,1)}%）`;}
      return `${context.dataset.label}：${fmt(context.parsed.y,4)}`;
    }}}},scales:{x:{type:'linear',min:Math.max(0,visibleMin-padding),max:visibleMax+padding,ticks:{color:colors.sub,callback:value=>`${fmt(Number(value),0)}%`,maxTicksLimit:9},grid:{color:colors.grid},title:{display:true,text:`HV${window} 年化波动率（%）`,color:colors.sub}},y:{beginAtZero:true,ticks:{color:colors.sub,maxTicksLimit:6},grid:{color:colors.grid},title:{display:true,text:'概率密度',color:colors.sub}}}};
    if(distributionChart)distributionChart.destroy();
    distributionChart=new Chart(byId('distributionChart'),{data,options});
  }
  function renderSeries(window,rows,median,p90){
    const labels=rows.map(row=>row.date);
    const data={labels,datasets:[
      {label:`HV${window}`,data:rows.map(row=>row.value),borderColor:colors.blue,backgroundColor:'rgba(77,196,255,.08)',fill:true,pointRadius:0,borderWidth:1.8,tension:.08},
      {label:'中位数',data:rows.map(()=>median),borderColor:colors.cyan,pointRadius:0,borderWidth:1.2,borderDash:[4,4]},
      {label:'P90',data:rows.map(()=>p90),borderColor:colors.orange,pointRadius:0,borderWidth:1.2,borderDash:[6,4]}
    ]};
    const options={responsive:true,maintainAspectRatio:false,animation:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:colors.sub,usePointStyle:true,boxWidth:9}},tooltip:{callbacks:{label:context=>`${context.dataset.label}：${fmt(context.parsed.y,2)}%`}}},scales:{x:{ticks:{color:colors.sub,maxTicksLimit:9,maxRotation:0},grid:{color:colors.grid},title:{display:true,text:'日期',color:colors.sub}},y:{beginAtZero:true,ticks:{color:colors.sub,callback:value=>`${value}%`},grid:{color:colors.grid},title:{display:true,text:`HV${window}（%）`,color:colors.sub}}}};
    if(seriesChart)seriesChart.destroy();
    seriesChart=new Chart(byId('seriesChart'),{type:'line',data,options});
  }
  function render(){
    if(!payload)return;
    const window=Number(byId('horizon').value),rows=filteredRows(window);
    if(rows.length<20){byId('status').textContent='所选区间有效样本不足20个';byId('status').className='status bad';return;}
    byId('status').textContent=`${rows.length.toLocaleString()}个滚动观测`;byId('status').className='status';
    const values=rows.map(row=>row.value),stats=histogram(values),current=values[values.length-1];
    const median=quantile(stats.sorted,.5),p75=quantile(stats.sorted,.75),p90=quantile(stats.sorted,.9),p95=quantile(stats.sorted,.95),currentRank=percentileRank(stats.sorted,current);
    const atm=readNumber('atmIv'),put=readNumber('putIv');
    byId('statsKpis').innerHTML=[
      kpi(`区间末HV${window}`,`${fmt(current,2)}%`,`${rows[rows.length-1].date} · 历史${pctText(currentRank)}`,'red'),
      kpi('中位数',`${fmt(median,2)}%`,`均值 ${fmt(stats.avg,2)}%`,'blue'),
      kpi('P75',`${fmt(p75,2)}%`,`四分之三的历史观测不高于此值`,'green'),
      kpi('P90 / P95',`${fmt(p90,2)}% / ${fmt(p95,2)}%`,'历史高波动区参考','orange'),
      kpi('ATM IV分位',atm>0?pctText(percentileRank(stats.sorted,atm)):'未填写',atm>0?`${fmt(atm,2)}%相对HV${window}历史分布`:'可在上方输入当前ATM IV','blue'),
      kpi('Put IV分位',put>0?pctText(percentileRank(stats.sorted,put)):'未填写',put>0?`${fmt(put,2)}%相对HV${window}历史分布`:'可在上方输入目标Put IV','green')
    ].join('');
    byId('distributionTitle').textContent=`HV${window}分布`;byId('seriesTitle').textContent=`HV${window}时间序列`;
    byId('selectionSummary').innerHTML=`统计区间 <b>${rows[0].date}—${rows[rows.length-1].date}</b>。区间末HV${window}为 <b>${fmt(current,2)}%</b>，处于历史 <b>${pctText(currentRank)}</b>；分布中位数 <b>${fmt(median,2)}%</b>。`;
    renderDistribution(window,rows,stats);renderSeries(window,rows,median,p90);
  }
  async function init(){
    try{
      const response=await fetch('hs300_volatility_data.json',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
      payload=await response.json();byId('latestDate').textContent=`最新收盘 ${payload.last_date}`;
      ['startDate','endDate'].forEach(id=>{byId(id).min=payload.first_date;byId(id).max=payload.last_date;});
      byId('startDate').value=payload.first_date;byId('endDate').value=payload.last_date;
      document.querySelectorAll('#horizon,#startDate,#endDate,#atmIv,#putIv').forEach(input=>input.addEventListener('input',render));render();
    }catch(error){byId('status').textContent='数据加载失败';byId('status').className='status bad';byId('selectionSummary').textContent=`无法读取波动率数据：${error.message}`;}
  }
  init();
})();
