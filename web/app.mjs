import {runFusion} from './fusion.mjs';
const $=id=>document.getElementById(id), state={visible:null,infrared:null}, urls={};
let busy=false, generation=0, report='';
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function invalidate(){generation++;report='';$('results').hidden=true;}
function enable(){
  $('run').disabled=busy||!state.visible||!state.infrared;
  document.querySelectorAll('.sample,input').forEach(element=>{element.disabled=busy;});
}
function release(key){if(urls[key])URL.revokeObjectURL(urls[key]);}
async function decode(blob,name){
  if(!blob.size||blob.size>16*1024*1024)throw Error('文件需非空且不超过 16 MiB。');
  const bitmap=await createImageBitmap(blob),{width,height}=bitmap;
  if(width<2||height<2||width*height>4000000){bitmap.close();throw Error('图像需为 2×2 至 400 万像素。');}
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(bitmap,0,0);bitmap.close();
  const rgba=context.getImageData(0,0,width,height).data,data=new Uint8Array(width*height);
  for(let n=0;n<data.length;n++){
    if(rgba[n*4+3]!==255)throw Error('暂不支持透明图像，请转换为不带透明通道的 8-bit 图片。');
    data[n]=Math.round(.299*rgba[n*4]+.587*rgba[n*4+1]+.114*rgba[n*4+2]);
  }
  const digest=await crypto.subtle.digest('SHA-256',await blob.arrayBuffer());
  const sha256=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
  return {width,height,data,name,sha256,blob};
}
function showInput(key,image){
  state[key]=image;release(key);urls[key]=URL.createObjectURL(image.blob);
  $(key+'-preview').src=urls[key];$(key+'-preview').hidden=false;$(key+'-empty').hidden=true;
  $(key+'-info').textContent=`${image.name} · ${image.width} × ${image.height}`;
}
for(const key of ['visible','infrared'])$(key+'-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  invalidate();const ticket=generation;busy=true;enable();status('正在读取图片…');
  try{const image=await decode(file,file.name);if(ticket!==generation)return;showInput(key,image);status('图片已载入，点击运行融合。');}
  catch(error){if(ticket===generation){state[key]=null;$(key+'-preview').hidden=true;$(key+'-empty').hidden=false;$(key+'-info').textContent='读取失败';status(error.message,true);}}
  finally{if(ticket===generation){busy=false;enable();}}
});
for(const button of document.querySelectorAll('[data-sample]'))button.addEventListener('click',async()=>{
  invalidate();const ticket=generation;busy=true;enable();status('正在载入样例…');
  const name=button.dataset.sample;
  try{
    const images=await Promise.all(['visible','infrared'].map(async key=>{
      const path=name==='synthetic'?`samples/${key}.png`:`samples/tno/${key}/${name}.png`;
      const response=await fetch(path);if(!response.ok)throw Error('样例加载失败，请重试。');
      return decode(await response.blob(),`${name}-${key}.png`);
    }));
    if(ticket!==generation)return;
    showInput('visible',images[0]);showInput('infrared',images[1]);
    document.querySelectorAll('.sample').forEach(el=>el.classList.toggle('selected',el===button));
    status('样例已载入，点击“运行两种融合方法”。');
  }catch(error){if(ticket===generation)status(error.message,true);}
  finally{if(ticket===generation){busy=false;enable();}}
});
$('weight').addEventListener('input',()=>{$('weight-value').textContent=Number($('weight').value).toFixed(2);invalidate();});
$('task').addEventListener('input',invalidate);
function renderImage(name,image){
  const canvas=$(name+'-canvas');canvas.width=image.width;canvas.height=image.height;
  const context=canvas.getContext('2d'),rgba=context.createImageData(image.width,image.height);
  for(let n=0;n<image.data.length;n++){rgba.data[n*4]=rgba.data[n*4+1]=rgba.data[n*4+2]=image.data[n];rgba.data[n*4+3]=255;}
  context.putImageData(rgba,0,0);$(name+'-download').href=canvas.toDataURL('image/png');
}
$('run').addEventListener('click',async()=>{
  if(busy)return;busy=true;enable();status('正在融合与计算指标…');const ticket=generation;
  await new Promise(resolve=>setTimeout(resolve,30));
  try{
    if(ticket!==generation)return;
    const start=performance.now(),weight=Number($('weight').value),out=runFusion(state.visible,state.infrared,weight);
    $('metric-rows').replaceChildren();
    const lines=['# FusionAgent browser experiment','','Mode: deterministic-browser',`Task: ${$('task').value.replace(/[<>]/g,'')}`,
      `Visible weight: ${weight}`,'','## Inputs',...['visible','infrared'].map(key=>`${key}: ${state[key].name.replace(/[<>\r\n]/g,'')} (${state[key].width}x${state[key].height}), SHA256 ${state[key].sha256}`),'',
      '## Metrics','| Method | Entropy | Std | Avg gradient | Spatial frequency |','|---|---:|---:|---:|---:|'];
    for(const name of ['weighted','wavelet']){
      const item=out.results[name];renderImage(name,item.image);
      const row=document.createElement('tr'),values=Object.values(item.metrics).map(v=>v.toFixed(4));
      for(const value of [name,...values]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
      $('metric-rows').append(row);lines.push(`| ${name} | ${values.join(' | ')} |`);
    }
    const explanation=`${out.recommendation} 的平均梯度更高（平局选 weighted）。这只是细节启发式，不代表整体质量更好，请同时检查噪声和目标保留。`;
    $('recommendation').textContent=explanation;$('trace').textContent=JSON.stringify(out.trace,null,2);
    report=[...lines,'','## Recommendation',explanation,'','## Executed tools',JSON.stringify(out.trace,null,2),'','## Limits',
      'Browser decoding and grayscale conversion may differ from OpenCV. Haar roundoff may differ by 1 intensity level. No model, registration or reference metrics.',
      'Task text is a note only. Processing is local to this browser. Higher metrics do not prove quality.',
      'TNO examples: Alexander Toet / TNO, CC BY 4.0, https://doi.org/10.6084/m9.figshare.1008029.v2'].join('\n');
    $('results').hidden=false;status(`完成 · ${(performance.now()-start).toFixed(0)} ms · 图片未上传到服务器`);
  }catch(error){$('results').hidden=true;status(error.message,true);}
  finally{busy=false;enable();}
});
$('download-report').addEventListener('click',()=>{
  if(!report)return;release('report');urls.report=URL.createObjectURL(new Blob([report],{type:'text/markdown;charset=utf-8'}));
  const anchor=document.createElement('a');anchor.href=urls.report;anchor.download='fusion-report.md';anchor.click();
});
