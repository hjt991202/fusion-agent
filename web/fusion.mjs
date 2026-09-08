/** Browser CPU baselines, matching the Python definitions (no network or model calls). */
export function validate(image) {
  if (!image || !Number.isInteger(image.width) || !Number.isInteger(image.height) ||
      image.width < 2 || image.height < 2 || image.width * image.height > 4000000 ||
      image.data.length !== image.width * image.height) throw Error('图像需为 2×2 至 400 万像素。');
}
function pair(v, i) {
  validate(v); validate(i);
  if (v.width !== i.width || v.height !== i.height) throw Error('两张图尺寸不一致。请上传已配准、同尺寸的图像对。');
}
export function roundEven(x) {
  const base = Math.floor(x), delta = x - base;
  return delta === 0.5 ? base + (base % 2) : Math.round(x);
}
const byte = x => Math.min(255, Math.max(0, roundEven(x)));
export function weighted(v, i, weight = 0.5) {
  pair(v, i);
  if (!Number.isFinite(weight) || weight < 0 || weight > 1) throw Error('权重应在 0 到 1 之间。');
  return {...v, data: Uint8Array.from(v.data, (x, n) => byte(weight * x + (1 - weight) * i.data[n]))};
}
export function wavelet(v, i) {
  pair(v, i);
  const w = v.width, h = v.height, out = new Uint8Array(w * h);
  const transform = (p, x, y) => {
    const x1 = Math.min(x + 1, w - 1), y1 = Math.min(y + 1, h - 1);
    const a = p[y*w+x], b = p[y*w+x1], c = p[y1*w+x], d = p[y1*w+x1];
    return [(a+b+c+d)/2, (a+b-c-d)/2, (a-b+c-d)/2, (a-b-c+d)/2];
  };
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    const a = transform(v.data,x,y), b = transform(i.data,x,y);
    const [ll, lh, hl, hh] = a.map((value,k) => k === 0 ? (value+b[k])/2 : Math.abs(value) >= Math.abs(b[k]) ? value : b[k]);
    out[y*w+x] = byte((ll+lh+hl+hh)/2);
    if (x+1 < w) out[y*w+x+1] = byte((ll+lh-hl-hh)/2);
    if (y+1 < h) out[(y+1)*w+x] = byte((ll-lh+hl-hh)/2);
    if (x+1 < w && y+1 < h) out[(y+1)*w+x+1] = byte((ll-lh-hl+hh)/2);
  }
  return {width:w,height:h,data:out};
}
export function metrics(image) {
  validate(image);
  const {width:w,height:h,data:p} = image, n=p.length, bins=new Uint32Array(256);
  let sum=0, variance=0, gx=0, gy=0, gradient=0, entropy=0;
  for (const value of p) { bins[value]++; sum+=value; }
  const mean=sum/n;
  for (const value of p) variance+=(value-mean)**2;
  for (const count of bins) if (count) entropy-=count/n*Math.log2(count/n);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    const k=y*w+x;
    const dx=x+1<w?p[k+1]-p[k]:0, dy=y+1<h?p[k+w]-p[k]:0;
    if (x+1<w) gx+=dx*dx;
    if (y+1<h) gy+=dy*dy;
    if (x+1<w && y+1<h) gradient+=Math.sqrt((dx*dx+dy*dy)/2);
  }
  return {entropy,standard_deviation:Math.sqrt(variance/n),
    average_gradient:gradient/((w-1)*(h-1)),spatial_frequency:Math.sqrt(gx/(h*(w-1))+gy/((h-1)*w))};
}
export function runFusion(v,i,weight=0.5) {
  const results={}, trace=[];
  for (const [name,fn] of [['weighted',()=>weighted(v,i,weight)],['wavelet',()=>wavelet(v,i)]]) {
    const start=performance.now(), image=fn();
    trace.push({tool:name+'_fusion',milliseconds:performance.now()-start});
    const metricStart=performance.now(), measured=metrics(image);
    trace.push({tool:'calculate_metrics',method:name,milliseconds:performance.now()-metricStart});
    results[name]={image,metrics:measured};
  }
  return {results,trace,recommendation:results.wavelet.metrics.average_gradient>
    results.weighted.metrics.average_gradient?'wavelet':'weighted'};
}
