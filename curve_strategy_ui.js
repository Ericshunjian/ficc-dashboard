window.FiccUI=(()=>{
'use strict';const charts={},colors={bayes_temporal_new:'#16847b',ridge_temporal:'#5470c6',ridge_original:'#a6afbd'};
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%',side=v=>v===1?'做多·走扩':v===-1?'做空·收窄':'空仓';
function chart(id,dates,series,{type='line',y={},position=false}={}){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart(document.getElementById(id),{type,data:{labels:dates,datasets:series.map((s,i)=>({borderWidth:1.7,pointRadius:0,spanGaps:false,stepped:position,...s,borderColor:s.color||colors[s.key]||['#5470c6','#16847b'][i],backgroundColor:s.color||colors[s.key]||['#5470c6','#16847b'][i]}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{autoSkip:false,maxRotation:0,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=Math.min(b-a+1,this.width<450?3:5),keep=n<2?[a]:Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}},y:position?{min:-1.2,max:1.2,ticks:{stepSize:1}}:y}}});
 document.getElementById(id).ondblclick=()=>charts[id].resetZoom();return charts[id];
}
async function read(name){const r=await fetch(name+'?v='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(name+' '+r.status);return r.json();}
function reset(){Object.values(charts).forEach(c=>c.resetZoom());}
return {charts,colors,fmt,pct,side,chart,read,reset};
})();
