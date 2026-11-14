// 按 HTML 元素 ID 查找节点；后续统一使用这个简写。
const $ = id => document.getElementById(id);
// 保存最近一次成功的评分结果，供 CSV 导出使用。
let rows = [];
// 把后端特征名翻译成页面标签；键名必须与 schema.py 一致。
const labels = {
  usage_trend: '流量变化趋势', complaints_30d: '近30天投诉次数',
  active_days_30d: '近30天活跃天数', usage_mb_30d: '近30天流量',
  usage_mb_90d: '近90天流量', tenure_days: '在网时长', monthly_fee: '套餐月费',
  contract_days_remaining: '合约剩余天数', call_minutes_30d: '近30天通话时长',
  complaints_90d: '近90天投诉次数', overdue_invoices: '逾期账单数',
  bill_amount_90d: '近90天账单金额', plan_type: '套餐类型', region: '区域'
};

// 统一处理 JSON 请求、当前页面中的访问密钥和 HTTP 错误。
async function api(path, options = {}) {
  // 展开调用方参数，同时为请求添加 JSON 类型和 API 密钥头。
  const response = await fetch(path, {
    ...options,
    headers: {'Content-Type': 'application/json', 'X-API-Key': $('api-key').value, ...options.headers}
  });
  // 即使是错误响应，也先读取 JSON，以便展示 FastAPI 的 detail。
  const data = await response.json();
  // 非 2xx 状态转成异常，由页面事件处理函数显示。
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}

// 创建文本单元格；使用 textContent，避免把数据当作 HTML 执行。
function cell(row, text, cls = '') {
  const td = document.createElement('td'); // 创建一个尚未加入页面的单元格。
  td.textContent = text; // 写入显示文本。
  td.className = cls; // 添加可选样式，例如多行的业务信号列。
  row.append(td); // 将单元格挂到当前表格行。
  return td; // 风险等级列还需要在单元格内追加彩色标签。
}

// 连接模型并展示测试指标和验证集置换重要性。
async function connect() {
  $('connect').disabled = true; // 防止用户在请求完成前重复连接。
  try {
    const meta = await api('/api/v1/model'); // 获取当前已加载模型的元数据。
    $('status').textContent = '● 模型已就绪'; // 更新连接状态。
    $('version').textContent = meta.model_version; // 展示实际服务版本。
    $('algorithm').textContent = meta.selected_algorithm; // 展示验证集选出的算法。
    const m = meta.test_metrics; // 以下卡片统一使用独立测试集指标。
    $('auc').textContent = m.roc_auc?.toFixed(3) ?? '—'; // 单类无法定义 AUC 时显示占位符。
    $('ap').textContent = m.average_precision?.toFixed(3) ?? '—'; // 展示 AP，保留三位小数。
    $('lift').textContent = m.lift_at_k ? `${m.lift_at_k.toFixed(2)}×` : '—'; // 展示 Top-K 提升倍数。
    $('capacity').textContent = `风险排名前 ${(m.top_fraction * 100).toFixed(0)}%`; // 说明 Top-K 比例。
    $('importance').replaceChildren(); // 清理上次模型的特征条形图。
    const top = meta.permutation_importance.slice(0, 7); // 只显示排名最靠前的七个特征。
    const max = Math.max(...top.map(x => x.mean), 0.00001); // 计算归一化宽度并避免零分母。
    top.forEach(item => {
      const wrapper = document.createElement('div'); // 创建单个特征的容器。
      wrapper.className = 'bar-row'; // 应用条形图间距。
      const label = document.createElement('div'); // 创建名称和数值所在行。
      label.className = 'bar-label'; // 让名称和数值两端对齐。
      const name = document.createElement('span'); // 创建名称节点。
      name.textContent = labels[item.feature] || item.feature; // 未翻译特征保留原名。
      const value = document.createElement('span'); // 创建数值节点。
      value.textContent = item.mean.toFixed(4); // 显示平均 AP 降幅。
      label.append(name, value); // 把两个文本节点放进标签行。
      const track = document.createElement('div'); // 创建条形图背景轨道。
      track.className = 'bar-track'; // 应用轨道样式。
      const fill = document.createElement('div'); // 创建实际条形。
      fill.className = 'bar-fill'; // 应用填充颜色。
      fill.style.width = `${Math.max(0, item.mean) / max * 100}%`; // 负重要性保留数值，但条宽截为零。
      track.append(fill); // 把填充放入轨道。
      wrapper.append(label, track); // 组合标签和轨道。
      $('importance').append(wrapper); // 将当前特征插入页面。
    });
    $('message').textContent = ''; // 成功连接后清除旧错误。
  } catch (e) {
    $('status').textContent = '连接未完成'; // 标记连接失败。
    $('message').textContent = e.message; // 展示服务返回的错误原因。
  } finally {
    $('connect').disabled = false; // 成败均恢复按钮。
  }
}

// 连接按钮允许用户在更换密钥后重新加载模型信息。
$('connect').addEventListener('click', connect);
// 选择本地 JSON 文件后，把内容载入文本框；文件此时尚未提交给服务器。
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; // 获取用户选择的第一个文件。
  if (!f) return; // 取消文件选择时不继续。
  if (f.size > 2_000_000) { // 限制页面导入文件大小。
    $('message').textContent = '文件大小不能超过 2 MB';
    return;
  }
  $('payload').value = await f.text(); // 异步读取文件文本并填入输入框。
});

// 点击评分时清理旧结果，提交当前输入，并按风险降序展示。
$('predict').addEventListener('click', async () => {
  $('predict').disabled = true; // 请求期间禁止重复提交。
  $('message').textContent = '正在计算…'; // 告知用户请求正在进行。
  rows = []; // 清除上一批数据，避免导出旧结果。
  $('download').disabled = true; // 新结果到达前禁止导出。
  $('count').textContent = ''; // 清除旧人数。
  $('results').replaceChildren(); // 清空旧表格行。
  try {
    const payload = JSON.parse($('payload').value); // 非法 JSON 会进入 catch。
    rows = await api('/api/v1/predict', {method: 'POST', body: JSON.stringify(payload)}); // 发送客户特征。
    rows.sort((a, b) => b.risk_score - a.risk_score); // 将高分客户放在前面。
    rows.forEach(item => {
      const tr = document.createElement('tr'); // 为一位客户创建一行。
      cell(tr, item.customer_id); // 第一列显示客户编号。
      cell(tr, item.snapshot_date); // 第二列显示评分日期。
      cell(tr, (item.risk_score * 100).toFixed(1) + '%'); // 将风险分数格式化为百分数。
      const risk = cell(tr, ''); // 预留风险标签所在单元格。
      const badge = document.createElement('span'); // 创建彩色等级标签。
      badge.className = `risk ${item.risk_level}`; // high/medium/low 对应不同 CSS 配色。
      badge.textContent = {high: '高风险', medium: '中风险', low: '低风险'}[item.risk_level]; // 翻译等级。
      risk.append(badge); // 插入标签。
      cell(tr, item.risk_signals.join(' · ') || '—', 'signals'); // 合并规则信号，无命中时显示占位符。
      cell(tr, item.suggested_action); // 显示建议回访动作。
      $('results').append(tr); // 把客户行追加到表格。
    });
    $('count').textContent = `${rows.length} 位`; // 显示返回客户数。
    $('message').textContent = '评分完成'; // 提示请求成功。
    $('download').disabled = false; // 允许导出刚得到的结果。
  } catch (e) {
    $('message').textContent = e.message; // 展示 JSON 或接口错误。
  } finally {
    $('predict').disabled = false; // 无论成败均允许再次提交。
  }
});

// 在浏览器中生成 CSV，不需要额外调用后端。
$('download').addEventListener('click', () => {
  const fields = ['customer_id', 'snapshot_date', 'risk_score', 'risk_level', 'risk_signals', 'suggested_action', 'model_version']; // 固定导出列顺序。
  const quote = value => {
    let text = Array.isArray(value) ? value.join('; ') : String(value); // 列表转成分号分隔文本。
    if (/^[=+@\-\t\r]/.test(text)) text = "'" + text; // 防止电子表格把用户文本当作公式执行。
    return '"' + text.replaceAll('"', '""') + '"'; // 双引号翻倍后包裹字段，兼容逗号和换行。
  };
  const csv = '\uFEFF' + [fields.join(','), ...rows.map(row => fields.map(key => quote(row[key])).join(','))].join('\r\n'); // 添加 UTF-8 BOM，便于表格软件识别中文。
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8;'})); // 为内存文件创建临时下载地址。
  const a = document.createElement('a'); // 创建下载链接。
  a.href = url; // 指向内存 CSV。
  a.download = 'customer-risk-scores.csv'; // 设置下载文件名。
  a.click(); // 触发浏览器下载。
  setTimeout(() => URL.revokeObjectURL(url), 1000); // 稍后释放临时地址所占资源。
});

connect(); // 页面加载完成后立即尝试连接模型。
