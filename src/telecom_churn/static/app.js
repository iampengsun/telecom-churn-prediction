const $ = id => document.getElementById(id);
let rows = [];
const labels = {usage_trend:'流量变化趋势',complaints_30d:'近30天投诉次数',active_days_30d:'近30天活跃天数',usage_mb_30d:'近30天流量',usage_mb_90d:'近90天流量',tenure_days:'在网时长',monthly_fee:'套餐月费',contract_days_remaining:'合约剩余天数',call_minutes_30d:'近30天通话时长',complaints_90d:'近90天投诉次数',overdue_invoices:'逾期账单数',bill_amount_90d:'近90天账单金额',plan_type:'套餐类型',region:'区域'};
async function api(path, options={}) {
  const response = await fetch(path, {...options, headers:{'Content-Type':'application/json','X-API-Key':$('api-key').value,...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function cell(row, text, cls='') {const td=document.createElement('td');td.textContent=text;td.className=cls;row.append(td);return td;}
async function connect() {
  $('connect').disabled=true;
  try {
    const meta=await api('/api/v1/model');
    $('status').textContent='● 模型已就绪';$('version').textContent=meta.model_version;
    $('algorithm').textContent=meta.selected_algorithm;
    const m=meta.test_metrics;
    $('auc').textContent=m.roc_auc?.toFixed(3) ?? '—';$('ap').textContent=m.average_precision?.toFixed(3) ?? '—';
    $('lift').textContent=m.lift_at_k ? `${m.lift_at_k.toFixed(2)}×` : '—';
    $('capacity').textContent=`风险排名前 ${(m.top_fraction*100).toFixed(0)}%`;
    $('importance').replaceChildren();
    const top=meta.permutation_importance.slice(0,7);const max=Math.max(...top.map(x=>x.mean),0.00001);
    top.forEach(item=>{const wrapper=document.createElement('div');wrapper.className='bar-row';const label=document.createElement('div');label.className='bar-label';const name=document.createElement('span');name.textContent=labels[item.feature]||item.feature;const value=document.createElement('span');value.textContent=item.mean.toFixed(4);label.append(name,value);const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${Math.max(0,item.mean)/max*100}%`;track.append(fill);wrapper.append(label,track);$('importance').append(wrapper);});
    $('message').textContent='';
  } catch(e) {$('status').textContent='连接未完成';$('message').textContent=e.message;}
  finally {$('connect').disabled=false;}
}
$('connect').addEventListener('click',connect);
$('file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;if(f.size>2_000_000){$('message').textContent='文件大小不能超过 2 MB';return;}$('payload').value=await f.text();});
$('predict').addEventListener('click',async()=>{
  $('predict').disabled=true;$('message').textContent='正在计算…';rows=[];$('download').disabled=true;$('count').textContent='';$('results').replaceChildren();
  try {
    const payload=JSON.parse($('payload').value);
    rows=await api('/api/v1/predict',{method:'POST',body:JSON.stringify(payload)});
    rows.sort((a,b)=>b.risk_score-a.risk_score);
    rows.forEach(item=>{const tr=document.createElement('tr');cell(tr,item.customer_id);cell(tr,item.snapshot_date);cell(tr,(item.risk_score*100).toFixed(1)+'%');const risk=cell(tr,'');const badge=document.createElement('span');badge.className=`risk ${item.risk_level}`;badge.textContent={high:'高风险',medium:'中风险',low:'低风险'}[item.risk_level];risk.append(badge);cell(tr,item.risk_signals.join(' · ')||'—','signals');cell(tr,item.suggested_action);$('results').append(tr);});
    $('count').textContent=`${rows.length} 位`;$('message').textContent='评分完成';$('download').disabled=false;
  }catch(e){$('message').textContent=e.message;}finally{$('predict').disabled=false;}
});
$('download').addEventListener('click',()=>{
  const fields=['customer_id','snapshot_date','risk_score','risk_level','risk_signals','suggested_action','model_version'];
  const quote=value=>{let text=Array.isArray(value)?value.join('; '):String(value);if(/^[=+@\-\t\r]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
  const csv='\uFEFF'+[fields.join(','),...rows.map(row=>fields.map(key=>quote(row[key])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));const a=document.createElement('a');a.href=url;a.download='customer-risk-scores.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
connect();
