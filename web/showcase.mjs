const methods=[
  ['weighted','加权平均','weighted_average_fusion'],
  ['wavelet','Haar 小波','wavelet_fusion'],
  ['icig','ICIG','icig_fusion'],
  ['letters','letters','letters_fusion'],
];
const metricKeys=['entropy','standard_deviation','average_gradient','spatial_frequency'];

export function validateShowcase(manifest){
  if(!manifest||!Array.isArray(manifest.pairs)||manifest.pairs.length!==15){
    throw Error('演示清单需包含 15 对已验证样例。');
  }
  const ids=new Set();
  for(const pair of manifest.pairs){
    if(!pair||!/^jiandu_\d{4,}$/.test(pair.id)||ids.has(pair.id)||pair.preview!==`${pair.id}.png`){
      throw Error('演示清单包含无效或重复的样例路径。');
    }
    ids.add(pair.id);
    if(!Number.isSafeInteger(pair.source_id)||pair.source_id<0||
       !Number.isSafeInteger(pair.width)||pair.width<2||
       !Number.isSafeInteger(pair.height)||pair.height<2||
       !Number.isFinite(pair.seconds)||pair.seconds<0){
      throw Error('演示清单包含无效尺寸或耗时。');
    }
    if(!Array.isArray(pair.timings))throw Error('缺少实际执行时间记录。');
    for(const [key,,tool] of methods){
      if(metricKeys.some(metric=>!Number.isFinite(pair.metrics?.[key]?.[metric])||pair.metrics[key][metric]<0)){
        throw Error('演示清单缺少完整实测指标。');
      }
      const timing=pair.timings.find(item=>item.tool===tool);
      if(!timing||!Number.isFinite(timing.seconds)||timing.seconds<0){
        throw Error('演示清单缺少方法执行记录。');
      }
    }
  }
  return manifest.pairs;
}

export async function initializeShowcase({loadPair}){
  const $=id=>document.getElementById(id);
  const select=$('showcase-select'),load=$('showcase-load'),status=$('showcase-status');
  let selected=null;
  function show(pair){
    selected=pair;
    const preview=`showcase/${pair.preview}`;
    $('showcase-preview').src=preview;
    $('showcase-preview').alt=`简牍 ${pair.source_id}：可见光、红外、ICIG 与 letters 彩色融合及同位置局部细节`;
    $('showcase-preview-link').href=preview;
    $('showcase-visible').href=`samples/jiandu/visible/${pair.id}.png`;
    $('showcase-infrared').href=`samples/jiandu/infrared/${pair.id}.png`;
    $('showcase-metrics').replaceChildren();
    for(const [key,label,tool] of methods){
      const row=document.createElement('tr');
      const seconds=pair.timings.find(item=>item.tool===tool).seconds;
      const values=[label,...metricKeys.map(metric=>pair.metrics[key][metric].toFixed(4)),seconds.toFixed(3)];
      for(const value of values){
        const cell=document.createElement('td');cell.textContent=value;row.append(cell);
      }
      $('showcase-metrics').append(row);
    }
    status.textContent=`简牍 ${pair.source_id} · ${pair.width} × ${pair.height} 原始像素 · 该组四方法流程总耗时 ${pair.seconds.toFixed(2)} 秒`;
    status.classList.remove('error');
    $('showcase-content').hidden=false;
  }
  $('showcase-preview').addEventListener('error',()=>{
    status.textContent='结果预览加载失败；可重选样例或打开完整实测记录。';
    status.classList.add('error');
  });
  try{
    const response=await fetch('showcase/manifest.json');
    if(!response.ok)throw Error('暂时无法载入预计算结果。浏览器基线仍可使用。');
    const pairs=validateShowcase(await response.json());
    select.replaceChildren();
    for(const pair of pairs){
      const option=document.createElement('option');
      option.value=pair.id;option.textContent=`简牍 ${pair.source_id} · ${pair.width} × ${pair.height}`;
      select.append(option);
    }
    const initial=pairs.find(pair=>pair.source_id===843)||pairs[0];
    select.value=initial.id;show(initial);
    select.disabled=false;load.dataset.ready='true';load.disabled=false;
    select.addEventListener('change',()=>show(pairs.find(pair=>pair.id===select.value)));
    load.addEventListener('click',async()=>{
      if(!selected)return;
      const pair=selected;
      const loaded=await loadPair(pair.id,{
        visible:`samples/jiandu/visible/${pair.id}.png`,
        infrared:`samples/jiandu/infrared/${pair.id}.png`,
      });
      if(loaded)$('baseline-inputs').scrollIntoView({behavior:'smooth',block:'start'});
    });
  }catch(error){
    status.textContent=error.message;status.classList.add('error');
    select.replaceChildren();
    const option=document.createElement('option');option.textContent='预计算演示暂不可用';select.append(option);
    select.disabled=true;load.disabled=true;delete load.dataset.ready;
  }
}
