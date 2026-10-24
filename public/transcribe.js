/* Mojo Studios — Transcribe: turns a song's audio into sheet music. Loaded by index.html as <script src="transcribe.js">.
   Pipeline: decode → pitch track (YIN for a solo line, harmonic salience for a full mix) → notes →
   tempo/beat grid → quantised rhythm → a SONG_LIBRARY-style song that opens in the compose engine.
   The analysis core (window.Transcriber) is plain JS with no DOM use so it also runs under node for testing. */
(function(root){
  'use strict';
  const SR=22050, N=2048, HOP=256, FPS=SR/HOP;
  const hz2midi=f=>69+12*Math.log2(f/440);
  const median=a=>{if(!a.length)return NaN;const s=Float64Array.from(a).sort();const m=s.length>>1;return s.length&1?s[m]:(s[m-1]+s[m])/2;};
  const pct=(a,p)=>{if(!a.length)return 0;const s=Float64Array.from(a).sort();return s[Math.min(s.length-1,Math.floor(p*s.length))];};
  const tick=()=>new Promise(r=>setTimeout(r,0));

  /* ---------- FFT (in-place radix-2, complex) ---------- */
  function makeFFT(n){
    const rev=new Uint32Array(n), lg=Math.log2(n)|0;
    for(let i=0;i<n;i++){let r=0;for(let b=0;b<lg;b++)r|=((i>>b)&1)<<(lg-1-b);rev[i]=r;}
    const cos=new Float64Array(n/2), sin=new Float64Array(n/2);
    for(let i=0;i<n/2;i++){cos[i]=Math.cos(2*Math.PI*i/n);sin[i]=-Math.sin(2*Math.PI*i/n);}
    return function(re,im){
      for(let i=0;i<n;i++){const j=rev[i];if(j>i){let t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}}
      for(let size=2;size<=n;size<<=1){
        const half=size>>1, step=n/size;
        for(let i=0;i<n;i+=size)for(let j=0,k=0;j<half;j++,k+=step){
          const a=i+j,b=a+half, tr=re[b]*cos[k]-im[b]*sin[k], ti=re[b]*sin[k]+im[b]*cos[k];
          re[b]=re[a]-tr; im[b]=im[a]-ti; re[a]+=tr; im[a]+=ti;
        }
      }
    };
  }

  /* ---------- Frame analysis: spectrum peaks, energy, spectral flux, YIN ---------- */
  async function analyse(x,mode,onProgress){
    const nF=Math.max(1,Math.floor((x.length-N)/HOP)+1);
    const fft=makeFFT(N), fft2=makeFFT(2*N);
    const win=new Float64Array(N); for(let i=0;i<N;i++)win[i]=0.5-0.5*Math.cos(2*Math.PI*i/N);
    const re=new Float64Array(N), im=new Float64Array(N), re2=new Float64Array(2*N), im2=new Float64Array(2*N), ar=new Float64Array(2*N), ai=new Float64Array(2*N);
    const nb=N/2, prevLog=new Float64Array(nb);
    // spec8: log-magnitude spectrum up to ~5.5 kHz, one byte per bin, kept for melody-onset detection.
    const KB=512, spec8=new Uint8Array(nF*KB), rmsS=new Float64Array(nF);
    const rms=new Float64Array(nF), flux=new Float64Array(nF), f0=new Float64Array(nF), conf=new Float64Array(nF);
    // Harmonic-salience grid for full mixes: 10-cent bins from MIDI 36 (C2) to 96 (C7).
    const SAL_LO=36, SAL_BINS=600, sal=mode==='mix'?new Float32Array(nF*SAL_BINS):null;
    const minLag=Math.floor(SR/1400), maxLag=Math.ceil(SR/65), d=new Float64Array(maxLag+2);
    for(let t=0;t<nF;t++){
      const o=t*HOP; let e=0;
      for(let i=0;i<N;i++){const v=x[o+i];e+=v*v;re[i]=v*win[i];im[i]=0;}
      rms[t]=Math.sqrt(e/N);
      // Short (23 ms) energy at the frame centre: the long pitch window smears the silences between notes.
      let es=0; for(let i=N/2-256;i<N/2+256;i++)es+=x[o+i]*x[o+i]; rmsS[t]=Math.sqrt(es/512);
      fft(re,im);
      let fl=0; const mag=re; // reuse buffer for magnitudes
      for(let k=0;k<nb;k++){const m=Math.hypot(re[k],im[k]);mag[k]=m;const lm=Math.log1p(100*m);const df=lm-prevLog[k];if(df>0&&k>2)fl+=df;prevLog[k]=lm;if(k<KB)spec8[t*KB+k]=Math.min(255,Math.round(lm*23));}
      flux[t]=fl;
      if(mode==='mix'){
        // Melodia-style salience: every spectral peak votes for the f0s it could be a harmonic of.
        let mx=0; for(let k=2;k<nb;k++)if(mag[k]>mx)mx=mag[k];
        const thr=mx*0.01, base=t*SAL_BINS;
        for(let k=3;k<nb-1&&k<500;k++){
          const a=mag[k]; if(a<thr||a<mag[k-1]||a<mag[k+1])continue;
          const al=Math.log(mag[k-1]+1e-12), ac=Math.log(a+1e-12), ar=Math.log(mag[k+1]+1e-12);
          const den=al-2*ac+ar, dk=den<0?0.5*(al-ar)/den:0;
          const fp=(k+dk)*SR/N, amp=Math.sqrt(a);
          for(let h=1;h<=8;h++){
            const b=Math.round((hz2midi(fp/h)-SAL_LO)*10); if(b<0)break; if(b>=SAL_BINS)continue;
            const w=amp*Math.pow(0.8,h-1);
            for(let j=-10;j<=10;j++){const bb=b+j;if(bb<0||bb>=SAL_BINS)continue;const c=Math.cos(Math.PI*j/20);sal[base+bb]+=w*c*c;}
          }
        }
      }else{
        // YIN difference function. r(τ)=Σ_{j<W} x[j]x[j+τ] via FFT cross-correlation of the first W samples with the frame.
        const W=N-maxLag;
        for(let i=0;i<2*N;i++){re2[i]=i<N?x[o+i]:0;im2[i]=0;ar[i]=i<W?x[o+i]:0;ai[i]=0;}
        fft2(ar,ai); fft2(re2,im2);
        for(let i=0;i<2*N;i++){const r=ar[i]*re2[i]+ai[i]*im2[i], q=ar[i]*im2[i]-ai[i]*re2[i];re2[i]=r;im2[i]=-q;} // conj(A)·B, conjugated for inverse
        fft2(re2,im2); // forward FFT of the conjugate = conj(N·IFFT); real part is 2N·r(τ)
        let e0=0; for(let i=0;i<W;i++)e0+=x[o+i]*x[o+i];
        let eL=e0, run=0; d[0]=1;
        for(let tau=1;tau<=maxLag;tau++){
          eL+=x[o+tau+W-1]*x[o+tau+W-1]-x[o+tau-1]*x[o+tau-1];
          const dd=Math.max(0,e0+eL-2*re2[tau]/(2*N));
          run+=dd; d[tau]=run>0?dd*tau/run:1;
        }
        let best=-1;
        for(let tau=minLag;tau<maxLag;tau++){if(d[tau]<0.15){while(tau+1<maxLag&&d[tau+1]<d[tau])tau++;best=tau;break;}}
        if(best<0){let mv=1e9;for(let tau=minLag;tau<maxLag;tau++)if(d[tau]<mv){mv=d[tau];best=tau;}}
        const a=d[best-1], b=d[best], c=d[best+1], den=a-2*b+c;
        const tp=best+(den>0?0.5*(a-c)/den:0);
        f0[t]=SR/tp; conf[t]=1-Math.min(1,b);
      }
      if((t&255)===255){onProgress&&onProgress(0.1+0.6*t/nF);await tick();}
    }
    let pitch;
    if(mode==='mix')pitch=await trackSalience(sal,nF,SAL_BINS,SAL_LO,rms,onProgress);
    else{
      const loud=pct(rms,0.95);
      pitch=new Float64Array(nF);
      for(let t=0;t<nF;t++)pitch[t]=(conf[t]>0.8&&rms[t]>loud*0.06&&f0[t]>60&&f0[t]<1500)?hz2midi(f0[t]):NaN;
    }
    // A clear dip in short-window energy is a gap between notes, whatever the pitch tracker says.
    for(let t=0;t<nF;t++){
      let m=0; for(let k=Math.max(0,t-25);k<=Math.min(nF-1,t+25);k++)if(rmsS[k]>m)m=rmsS[k];
      if(rmsS[t]<0.2*m)pitch[t]=NaN;
    }
    // Melody strength: log energy at the tracked pitch's first harmonics. Its jumps mark re-struck notes.
    const hEnv=new Float64Array(nF);
    for(let t=0;t<nF;t++){
      const p=pitch[t]; if(p!==p)continue;
      const f=440*Math.pow(2,(p-69)/12); let sum=0;
      for(let h=1;h<=4;h++){const k=Math.round(h*f*N/SR);if(k+1>=KB)break;sum+=Math.max(spec8[t*KB+k-1],spec8[t*KB+k],spec8[t*KB+k+1])/23;}
      hEnv[t]=sum;
    }
    return{nF,rms,flux,pitch,hEnv};
  }

  /* Viterbi through the salience grid: favour smooth melodies, allow leaps at a cost. */
  async function trackSalience(sal,nF,B,LO,rms,onProgress){
    const S=B/2; // 20-cent states
    const sv=new Float32Array(nF*S); const fmax=new Float64Array(nF);
    // Lead melodies sit above the bass: gently discount candidates below G3.
    const reg=new Float64Array(S); for(let s=0;s<S;s++){const m=LO+s*0.2;reg[s]=m<55?0.45+0.55*(m-LO)/(55-LO):1;}
    for(let t=0;t<nF;t++){let m=0;for(let s=0;s<S;s++){const v=Math.max(sal[t*B+2*s],sal[t*B+2*s+1])*reg[s];sv[t*S+s]=v;if(v>m)m=v;}fmax[t]=m;}
    const ref=pct(fmax,0.9)||1;
    const W=20, JUMP=1.2, STEP=0.04;
    let V=new Float64Array(S), V2=new Float64Array(S);
    const bp=new Int16Array(nF*S);
    for(let s=0;s<S;s++)V[s]=Math.log(sv[s]/ref+1e-3);
    for(let t=1;t<nF;t++){
      let gb=-1e18,gi=0; for(let s=0;s<S;s++)if(V[s]>gb){gb=V[s];gi=s;}
      for(let s=0;s<S;s++){
        let b=gb-JUMP,bi=gi;
        const lo=Math.max(0,s-W),hi=Math.min(S-1,s+W);
        for(let q=lo;q<=hi;q++){const v=V[q]-STEP*Math.abs(q-s);if(v>b){b=v;bi=q;}}
        V2[s]=b+Math.log(sv[t*S+s]/ref+1e-3); bp[t*S+s]=bi;
      }
      const tmp=V;V=V2;V2=tmp;
      if((t&511)===511){onProgress&&onProgress(0.7+0.15*t/nF);await tick();}
    }
    let s=0,b=-1e18; for(let q=0;q<S;q++)if(V[q]>b){b=V[q];s=q;}
    const path=new Int16Array(nF); for(let t=nF-1;t>=0;t--){path[t]=s;if(t>0)s=bp[t*S+s];}
    // Voicing: the tracked line must be a strong part of its frame and the frame must not be near-silent.
    const loud=pct(rms,0.95), pitch=new Float64Array(nF);
    const pv=[]; for(let t=0;t<nF;t++)pv.push(sv[t*S+path[t]]);
    const vthr=pct(pv,0.5)*0.35;
    for(let t=0;t<nF;t++){
      const v=pv[t], ok=v>vthr&&v>=fmax[t]*0.5&&rms[t]>loud*0.05;
      pitch[t]=ok?LO+path[t]*0.2+0.05:NaN;
    }
    return pitch;
  }

  /* ---------- Onsets from spectral flux ---------- */
  function onsetEnvelope(flux){
    const n=flux.length, env=new Float64Array(n);
    const m=pct(flux,0.97)||1;
    for(let t=0;t<n;t++){
      let s=0,c=0; for(let k=Math.max(0,t-8);k<=Math.min(n-1,t+8);k++){s+=flux[k];c++;}
      env[t]=Math.max(0,(flux[t]-s/c)/m);
    }
    return env;
  }
  function pickOnsets(env){
    const out=[], n=env.length, thr=Math.max(0.08,pct(env,0.9)*0.5);
    let last=-99;
    for(let t=1;t<n-1;t++){
      if(env[t]>thr&&env[t]>=env[t-1]&&env[t]>env[t+1]&&t-last>=5){out.push(t);last=t;}
    }
    return out;
  }

  function melodyOnsets(h){
    const out=[], n=h.length; let last=-99;
    const vals=[]; for(let t=0;t<n;t++)if(h[t]>0)vals.push(h[t]);
    const thr=Math.max(0.8,pct(vals,0.5)*0.06);
    for(let t=4;t<n-1;t++){
      if(!(h[t]>0))continue;
      let lo=1e9; for(let k=t-4;k<t;k++)if(h[k]>0&&h[k]<lo)lo=h[k];
      const rise=h[t]-lo, riseN=h[t+1]-Math.min(lo,h[t]);
      if(lo<1e9&&rise>thr&&rise>=riseN&&t-last>=6){out.push(t);last=t;}
    }
    return out;
  }

  /* ---------- Pitch frames → note events (in frames) ---------- */
  function segmentNotes(pitch,onsets){
    const n=pitch.length;
    // Global tuning correction: recordings are often a few cents sharp or flat.
    let sx=0,sy=0; for(let t=0;t<n;t++){const p=pitch[t];if(p===p){const a=2*Math.PI*(p-Math.round(p));sx+=Math.cos(a);sy+=Math.sin(a);}}
    const tune=Math.atan2(sy,sx)/(2*Math.PI);
    const p=new Float64Array(n);
    for(let t=0;t<n;t++){
      // 5-frame median over voiced neighbours removes single-frame octave blips
      const w=[]; for(let k=t-2;k<=t+2;k++)if(k>=0&&k<n&&pitch[k]===pitch[k])w.push(pitch[k]);
      p[t]=(pitch[t]===pitch[t]&&w.length>=3)?median(w)-tune:NaN;
    }
    const MIN=5; // ~58ms
    const notes=[]; let cur=null, cand=null, candN=0, gap=0;
    const close=end=>{if(cur){cur.end=end;if(cur.end-cur.start>=MIN)notes.push(cur);cur=null;}};
    for(let t=0;t<n;t++){
      if(p[t]!==p[t]){gap++;if(gap>=3)close(t-gap+1);cand=null;candN=0;continue;}
      const r=Math.round(p[t]);
      if(cur&&gap>0&&gap<3){/* bridge tiny dropouts */}
      gap=0;
      if(!cur){cur={start:t,end:t+1,frames:[p[t]]};continue;}
      const curR=Math.round(median(cur.frames.slice(-12)));
      if(r===curR){cur.frames.push(p[t]);cand=null;candN=0;continue;}
      if(cand===r)candN++;else{cand=r;candN=1;}
      if(candN>=MIN){
        const st=t-candN+1; const kept=cur.frames.length-(candN-1);
        cur.frames=cur.frames.slice(0,Math.max(1,kept)); close(st);
        cur={start:st,end:t+1,frames:[]}; for(let k=st;k<=t;k++)if(p[k]===p[k])cur.frames.push(p[k]);
        cand=null;candN=0;
      }else cur.frames.push(cur.frames[cur.frames.length-1]);
    }
    close(n);
    // Split repeated same-pitch notes at strong onsets or clear energy dips.
    const out=[];
    for(const nt of notes){
      let s=nt.start;
      const cuts=onsets.filter(o=>o>nt.start+4&&o<nt.end-4);
      cuts.sort((a,b)=>a-b);
      for(const c of cuts){if(c-s>=MIN&&nt.end-c>=MIN){out.push({start:s,end:c,midi:0,src:nt});s=c;}}
      out.push({start:s,end:nt.end,midi:0,src:nt});
    }
    for(const o of out){
      const fr=[]; for(let t=o.start;t<o.end;t++)if(p[t]===p[t])fr.push(p[t]);
      o.midi=Math.round(median(fr.length?fr:o.src.frames)); delete o.src;
    }
    return{notes:out.filter(o=>o.midi===o.midi),tune};
  }

  /* ---------- Tempo and beat grid ---------- */
  function estimateBeat(env,notes,nF){
    const s=new Float64Array(nF);
    for(let t=0;t<nF;t++)s[t]=env[t];
    for(const nt of notes){for(let k=-2;k<=2;k++){const t=nt.start+k;if(t>=0&&t<nF)s[t]+=Math.exp(-k*k/2);}}
    let mean=0; for(let t=0;t<nF;t++)mean+=s[t]; mean/=nF; for(let t=0;t<nF;t++)s[t]-=mean;
    // Autocorrelation over 50–200 BPM with a gentle prior around 110 BPM.
    const lo=Math.floor(FPS*60/200), hi=Math.ceil(FPS*60/50);
    let best=0, bl=lo; const ac=new Float64Array(hi+2);
    for(let L=lo;L<=hi+1;L++){let a=0;for(let t=0;t+L<nF;t++)a+=s[t]*s[t+L];ac[L]=a/(nF-L);}
    for(let L=lo;L<=hi;L++){
      const bpm=60*FPS/L, prior=Math.exp(-0.5*Math.pow(Math.log2(bpm/110)/0.9,2));
      const sc=(ac[L]+0.5*(ac[2*L]||0)+0.25*(ac[Math.round(L/2)]||0))*prior;
      if(sc>best){best=sc;bl=L;}
    }
    // Refine period (fractional) and phase together by comb alignment.
    const pos=t=>Math.max(0,s[Math.round(t)]||0);
    let bestP=bl,bestPh=0,bestSc=-1;
    for(let P=bl*0.96;P<=bl*1.04;P+=0.05){
      for(let ph=0;ph<P;ph+=0.5){
        let sc=0; for(let t=ph;t<nF;t+=P)sc+=pos(t-1)*0.5+pos(t)+pos(t+1)*0.5;
        sc/=Math.max(1,(nF-ph)/P);
        if(sc>bestSc){bestSc=sc;bestP=P;bestPh=ph;}
      }
    }
    return{period:bestP,phase:bestPh,bpm:60*FPS/bestP};
  }

  /* ---------- Key detection (Krumhansl–Kessler) ---------- */
  const KK_MAJ=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
  const KK_MIN=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];
  const KEY_NAMES=['C','D♭','D','E♭','E','F','F♯','G','A♭','A','B♭','B'];
  const MIN_NAMES=['C','C♯','D','E♭','E','F','F♯','G','G♯','A','B♭','B'];
  function corr(a,b){const ma=a.reduce((x,y)=>x+y)/12,mb=b.reduce((x,y)=>x+y)/12;let n=0,da=0,db=0;for(let i=0;i<12;i++){n+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)**2;db+=(b[i]-mb)**2;}return n/Math.sqrt(da*db||1);}
  function detectKey(notes){
    const h=new Array(12).fill(0); for(const n of notes)h[((n.midi%12)+12)%12]+=Math.min(n.end-n.start,60);
    let best={score:-2,tonic:0,minor:false};
    for(let k=0;k<12;k++){
      const rot=i=>h[(i+k)%12], hv=[...Array(12)].map((_,i)=>rot(i));
      const a=corr(hv,KK_MAJ), b=corr(hv,KK_MIN);
      if(a>best.score)best={score:a,tonic:k,minor:false};
      if(b>best.score)best={score:b,tonic:k,minor:true};
    }
    const relMaj=best.minor?(best.tonic+3)%12:best.tonic;
    const flats=[5,10,3,8,1].includes(relMaj); // F, B♭, E♭, A♭, D♭ majors read better with flats
    best.name=(best.minor?MIN_NAMES[best.tonic]+' minor':KEY_NAMES[best.tonic]+' major');
    best.flats=flats;
    return best;
  }
  const SHARP=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  const FLAT=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'];
  const NEUTRAL=['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'];
  function spell(midi,key){
    const names=key.flats?FLAT:(key.minor||key.tonic!==0)?SHARP:NEUTRAL;
    return names[((midi%12)+12)%12]+(Math.floor(midi/12)-1);
  }

  /* ---------- Quantise to a rhythmic grid and build song spec ---------- */
  const DUR16={16:'w',12:'h.',8:'h',6:'q.',4:'q',3:'e.',2:'e',1:'s'};
  const TS_INFO={'4/4':{bar:16},'3/4':{bar:12},'2/4':{bar:8}};
  // Can a note value of v sixteenths start at this point in the bar and still read cleanly?
  function fits(v,inBar,bar,rest){
    if(inBar+v>bar)return false;
    switch(v){
      case 16:return inBar===0;
      case 12:return inBar===0||(bar===16&&inBar===4);
      case 8:case 6:return inBar%4===0;
      case 4:case 3:return inBar%2===0;
      case 2:return !rest||inBar%2===0; // rests realign to the eighth grid first
      default:return true;
    }
  }
  // Break a run of sixteenths into readable note values (tied together by the caller).
  function pieces(len,pos,bar,rest){
    const out=[];
    while(len>0){
      const inBar=pos%bar; let pick=1;
      for(const v of [16,12,8,6,4,3,2,1])if(v<=len&&fits(v,inBar,bar,rest)){pick=v;break;}
      out.push(pick);pos+=pick;len-=pick;
    }
    return out;
  }
  function quantise(res,o){
    const ts=TS_INFO[o.ts]||TS_INFO['4/4'];
    const bpm=o.bpm||res.beat.bpm;
    // The beat period may be overridden by the user; keep the detected phase.
    const P=60*FPS/bpm, g16=P/4;
    const unit=o.detail==='simple'?2:1;
    const ph=res.beat.phase%P;
    let ev=res.notes.map(n=>{
      let a=Math.round((n.start-ph)/g16/unit)*unit, b=Math.round((n.end-ph)/g16/unit)*unit;
      if(b<=a)b=a+unit; return{a,b,midi:n.midi+o.transpose};
    }).sort((x,y)=>x.a-y.a);
    // Merge collisions (two notes snapped to one slot: keep the longer), trim overlaps, close tiny gaps.
    const m=[]; for(const e of ev){const l=m[m.length-1];if(l&&e.a<=l.a){if(e.b-e.a>l.b-l.a){l.b=e.b;l.midi=e.midi;}continue;}m.push(e);}
    for(let i=0;i<m.length;i++){
      const nx=m[i+1]; if(!nx)continue;
      if(m[i].b>nx.a)m[i].b=nx.a;
      const gap=nx.a-m[i].b; if(gap>0&&gap<=Math.max(unit,Math.floor((m[i].b-m[i].a)/3)))m[i].b=nx.a;
    }
    if(!m.length)return null;
    // Choose which beat is the downbeat: the one carrying the most note starts / longest notes.
    const b16=4, beatsPerBar=ts.bar/b16;
    let bestOff=0,bestSc=-1;
    for(let off=0;off<beatsPerBar;off++){
      let sc=0; for(const e of m){const beatIdx=Math.floor(e.a/b16);if(((beatIdx-off)%beatsPerBar+beatsPerBar)%beatsPerBar===0&&e.a%b16===0)sc+=1+Math.min(4,(e.b-e.a)/b16);}
      // Most tunes start on a downbeat rather than with a pickup.
      if((((Math.floor(m[0].a/b16)-off)%beatsPerBar)+beatsPerBar)%beatsPerBar===0)sc+=0.1*m.length;
      if(sc>bestSc){bestSc=sc;bestOff=off;}
    }
    if(o.barShift)bestOff=((bestOff+o.barShift)%beatsPerBar+beatsPerBar)%beatsPerBar;
    const first=m[0].a;
    let origin=bestOff*b16; while(origin>first)origin-=ts.bar; while(origin+ts.bar<=first)origin+=ts.bar;
    for(const e of m){e.a-=origin;e.b-=origin;}
    // Fit to the staff: pick clef from the median pitch and shift whole octaves if far out of range.
    const med=median(m.map(e=>e.midi));
    let clef=med<57?'bass':'treble', oct=0;
    const center=clef==='treble'?71:50;
    while(med+oct*12>center+12)oct--; while(med+oct*12<center-12)oct++;
    const spec=[]; let pos=0;
    const push=(len,tok)=>{let first=true;for(const v of pieces(len,pos,ts.bar,tok==='R')){const t=[tok,DUR16[v]];if(!first&&tok!=='R')t.push('~');spec.push(t);first=false;pos+=v;}};
    const key=o.key;
    for(const e of m){
      if(e.a>pos)push(e.a-pos,'R');
      push(e.b-e.a,spell(e.midi+oct*12,key));
    }
    const tail=(ts.bar-pos%ts.bar)%ts.bar; if(tail)push(tail,'R');
    return{spec,clef,octShift:oct,bars:pos/ts.bar,notes:m.length};
  }

  /* ---------- Public API ---------- */
  async function transcribe(samples,opts,onProgress){
    opts=opts||{};
    const mode=opts.mode==='solo'?'solo':'mix';
    let x=samples;
    // Normalise peak level so thresholds behave the same for quiet and loud files.
    let pk=0; for(let i=0;i<x.length;i++){const a=Math.abs(x[i]);if(a>pk)pk=a;}
    if(pk>0&&pk!==1){x=new Float32Array(x.length);for(let i=0;i<x.length;i++)x[i]=samples[i]/pk;}
    if(x.length<N*2)throw new Error('That clip is too short — give me at least a second of music.');
    const a=await analyse(x,mode,onProgress);
    onProgress&&onProgress(0.88);
    const env=onsetEnvelope(a.flux);
    // Same-pitch notes are split where the melody's own harmonics are re-struck; in a solo line any onset counts.
    const onsets=melodyOnsets(a.hEnv);
    if(mode==='solo')for(const o of pickOnsets(env))if(!onsets.some(q=>Math.abs(q-o)<5))onsets.push(o);
    const seg=segmentNotes(a.pitch,onsets);
    if(mode==='mix'&&seg.notes.length>4){
      // Stray bass/accompaniment notes far below the tune are tracker slips, not melody.
      const med=median(seg.notes.map(n=>n.midi)); seg.notes=seg.notes.filter(n=>n.midi>=med-14);
    }
    if(seg.notes.length<2)throw new Error('I could not hear a clear melody in that audio. Try a cleaner recording or switch the source type.');
    const beat=estimateBeat(env,seg.notes,a.nF);
    const key=detectKey(seg.notes);
    onProgress&&onProgress(1);
    return{mode,fps:FPS,duration:samples.length/SR,pitch:a.pitch,notes:seg.notes,tuneCents:Math.round(seg.tune*100),beat,key};
  }
  function toSong(res,o){
    o=Object.assign({ts:'4/4',detail:'normal',transpose:0,barShift:0,title:'My Transcription'},o||{});
    o.bpm=Math.round(o.bpm||res.beat.bpm);
    o.key=res.key;
    const q=quantise(res,o); if(!q)return null;
    return{id:o.id||('tx-'+Date.now().toString(36)),cat:'mine',title:o.title,sub:'Transcribed · '+res.key.name,emoji:'🎧',
      ts:o.ts,tempo:o.bpm,clef:q.clef,key:{name:'C',acc:0,type:'#'},spec:q.spec,_tx:{bars:q.bars,notes:q.notes,octShift:q.octShift,key:res.key.name}};
  }
  root.Transcriber={SR,transcribe,toSong,_internals:{quantise,pieces,detectKey,spell,estimateBeat,segmentNotes}};
})(typeof window!=='undefined'?window:globalThis);

/* ---------- Transcribe screen (browser only) ---------- */
(function(){
  if(typeof document==='undefined')return;
  const T=window.Transcriber, $=id=>document.getElementById(id);
  // Cmp is a top-level const in index.html, so it is not on window.
  const app=()=>typeof Cmp!=='undefined'?Cmp:null;
  const STORE='melodia:transcriptions', MAX_SAVED=20;
  const st={file:null,url:null,res:null,mode:'mix',ts:'4/4',detail:'normal',bpm:0,shift:0,oct:0,rec:null};

  const loadSaved=()=>{try{const a=JSON.parse(localStorage.getItem(STORE)||'[]');return Array.isArray(a)?a:[];}catch(e){return[];}};
  const writeSaved=a=>{try{localStorage.setItem(STORE,JSON.stringify(a.slice(0,MAX_SAVED)));}catch(e){}};
  // Saved transcriptions live in the Play library too (category "Mine").
  function syncLibrary(){
    const lib=window.SONG_LIBRARY; if(!lib)return;
    for(let i=lib.length-1;i>=0;i--)if(lib[i].cat==='mine')lib.splice(i,1);
    lib.push(...loadSaved());
    if(app())app()._quickCache=null;
  }

  function msg(t,err){const m=$('tx-msg');m.textContent=t||'';m.classList.toggle('err',!!err);}
  function setProg(f){const p=$('tx-prog');if(f==null){p.hidden=true;return;}p.hidden=false;$('tx-prog-bar').style.width=Math.round(f*100)+'%';}
  function seg(id,key,cb){
    $(id).addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;
      st[key]=b.dataset.v;[...$(id).children].forEach(x=>x.classList.toggle('on',x===b));cb&&cb();});
  }

  function useFile(blob,name){
    st.file=blob; st.res=null; $('tx-result').hidden=true;
    if(st.url)URL.revokeObjectURL(st.url);
    st.url=URL.createObjectURL(blob);
    const a=$('tx-audio'); a.src=st.url; a.hidden=false;
    $('tx-drop-t').textContent=name; $('tx-drop-d').textContent='Tap to choose a different file';
    $('tx-go').disabled=false; msg('');
    const base=(name||'').replace(/\.[^.]+$/,'').replace(/[_-]+/g,' ').trim();
    $('tx-title').value=(base&&base!=='Recording'?base:'My Transcription').slice(0,60);
  }

  async function decode(blob,start,len){
    const AC=window.AudioContext||window.webkitAudioContext;
    const ac=new AC(); let buf;
    try{buf=await ac.decodeAudioData(await blob.arrayBuffer());}
    catch(e){throw new Error('That file could not be read as audio. Try MP3, WAV or M4A.');}
    finally{ac.close&&ac.close();}
    start=Math.max(0,Math.min(start,Math.max(0,buf.duration-1)));
    len=Math.max(1,Math.min(len,buf.duration-start));
    // Resample the chosen clip to the analysis rate and mix it down to mono in one go.
    const off=new OfflineAudioContext(1,Math.ceil(len*T.SR),T.SR);
    const src=off.createBufferSource(); src.buffer=buf; src.connect(off.destination); src.start(0,start,len);
    const out=await off.startRendering();
    return{samples:out.getChannelData(0),start,len,total:buf.duration};
  }

  async function run(){
    if(!st.file)return;
    const go=$('tx-go'); go.disabled=true; msg('Listening…'); setProg(0.02);
    try{
      const d=await decode(st.file,+$('tx-start').value||0,Math.min(300,+$('tx-len').value||60));
      setProg(0.1); msg('Finding the melody…');
      st.res=await T.transcribe(d.samples,{mode:st.mode},f=>setProg(f));
      st.res.clipStart=d.start;
      st.bpm=Math.round(st.res.beat.bpm); st.shift=0; st.oct=0;
      $('tx-bpm').value=st.bpm; $('tx-shift').textContent='0'; $('tx-oct').textContent='0';
      msg(d.total>d.start+d.len+0.5?`Transcribed ${Math.round(d.len)} s of ${Math.round(d.total)} s. Change "from" to do another part.`:'');
      $('tx-result').hidden=false; refresh();
      $('tx-result').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(e){msg(e.message||String(e),true);}
    finally{setProg(null);go.disabled=false;}
  }

  function currentSong(){
    if(!st.res)return null;
    return T.toSong(st.res,{ts:st.ts,detail:st.detail,bpm:st.bpm,barShift:st.shift,transpose:12*st.oct,title:($('tx-title').value||'My Transcription').trim()});
  }

  function refresh(){
    const song=currentSong(); if(!song)return;
    const r=st.res, x=song._tx;
    $('tx-stats').innerHTML=[[r.key.name,'Key'],[song.tempo+' bpm','Tempo'],[x.notes,'Notes'],[x.bars,'Bars']]
      .map(([b,l])=>`<div class="tx-stat"><b>${b}</b><span>${l}</span></div>`).join('');
    drawRoll();
  }

  // Piano roll: detected pitch (dots), notes (bars) and the beat grid the rhythm was snapped to.
  function drawRoll(){
    const r=st.res, cv=$('tx-roll'), wrap=cv.parentElement;
    const pxs=60, W=Math.max(wrap.clientWidth||300,Math.ceil(r.duration*pxs)), H=170, dpr=window.devicePixelRatio||1;
    cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+'px'; cv.style.height=H+'px';
    const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
    const ms=r.notes.map(n=>n.midi); let lo=Math.min(...ms)-3, hi=Math.max(...ms)+3; if(hi-lo<14){const c=(hi+lo)/2;lo=c-7;hi=c+7;}
    const fx=f=>f/r.fps*pxs, fy=m=>H-8-(m-lo)/(hi-lo)*(H-16);
    const P=60*r.fps/st.bpm, bpb={'4/4':4,'3/4':3,'2/4':2}[st.ts];
    for(let k=0,t=r.beat.phase%P;t<r.pitch.length;t+=P,k++){
      g.fillStyle=k%bpb===0?'rgba(255,255,255,.14)':'rgba(255,255,255,.05)'; g.fillRect(Math.round(fx(t)),0,1,H);
    }
    g.fillStyle='rgba(244,114,182,.35)';
    for(let t=0;t<r.pitch.length;t++){const p=r.pitch[t];if(p===p)g.fillRect(fx(t),fy(p)-1,1.5,2);}
    g.fillStyle='#f472b6';
    for(const n of r.notes){const y=fy(n.midi);g.beginPath();g.roundRect?g.roundRect(fx(n.start),y-4,Math.max(3,fx(n.end-n.start)-1),8,3):g.rect(fx(n.start),y-4,Math.max(3,fx(n.end-n.start)-1),8);g.fill();}
  }

  function openSheet(){
    const song=currentSong(); if(!song)return;
    delete song._tx;
    const saved=loadSaved(); saved.unshift(song); writeSaved(saved);
    syncLibrary(); renderSaved();
    if(app())app().openSong(song.id);
  }

  function renderSaved(){
    const a=loadSaved(), card=$('tx-saved-card'); card.hidden=!a.length;
    const esc=t=>String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    $('tx-saved').innerHTML=a.map(s=>`<div class="tx-saved-row"><button class="tx-saved-open" data-id="${esc(s.id)}">🎧 ${esc(s.title)}<small>${esc(s.sub||'')}</small></button><button class="tx-saved-del" data-del="${esc(s.id)}" aria-label="Delete">×</button></div>`).join('');
  }

  async function toggleRec(){
    const btn=$('tx-rec');
    if(st.rec){st.rec.stop();return;}
    if(!navigator.mediaDevices||!window.MediaRecorder){msg('Recording is not supported in this browser. Choose a file instead.',true);return;}
    let stream;
    try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});}
    catch(e){msg('Mojo needs microphone access to record.',true);return;}
    const chunks=[], rec=new MediaRecorder(stream);
    rec.ondataavailable=e=>{if(e.data&&e.data.size)chunks.push(e.data);};
    rec.onstop=()=>{
      stream.getTracks().forEach(t=>t.stop()); st.rec=null;
      btn.classList.remove('on'); $('tx-rec-l').textContent='Record with the mic';
      const blob=new Blob(chunks,{type:rec.mimeType||'audio/webm'});
      if(blob.size)useFile(blob,'Recording');
    };
    rec.start(); st.rec=rec; btn.classList.add('on'); $('tx-rec-l').textContent='Stop recording'; msg('Recording… sing, hum or play the tune.');
  }

  function init(){
    if(!$('view-transcribe'))return;
    syncLibrary(); renderSaved();
    $('tx-file').addEventListener('change',e=>{const f=e.target.files&&e.target.files[0];if(f)useFile(f,f.name);});
    const drop=$('tx-drop');
    drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});
    drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
    drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');const f=e.dataTransfer.files&&e.dataTransfer.files[0];if(f)useFile(f,f.name);});
    $('tx-rec').addEventListener('click',toggleRec);
    $('tx-go').addEventListener('click',run);
    $('tx-open').addEventListener('click',openSheet);
    seg('tx-mode','mode'); seg('tx-ts','ts',refresh); seg('tx-detail','detail',refresh);
    $('tx-bpm').addEventListener('change',()=>{st.bpm=Math.max(30,Math.min(260,Math.round(+$('tx-bpm').value||st.bpm)));$('tx-bpm').value=st.bpm;refresh();});
    const setBpm=v=>{st.bpm=Math.max(30,Math.min(260,Math.round(v)));$('tx-bpm').value=st.bpm;refresh();};
    $('tx-half').addEventListener('click',()=>setBpm(st.bpm/2));
    $('tx-dbl').addEventListener('click',()=>setBpm(st.bpm*2));
    $('tx-result').addEventListener('click',e=>{
      const b=e.target.closest('.tx-step button'); if(!b)return; const d=+b.dataset.d;
      if(b.dataset.k==='bpm')setBpm(st.bpm+d);
      if(b.dataset.k==='shift'){st.shift+=d;$('tx-shift').textContent=st.shift;refresh();}
      if(b.dataset.k==='oct'){st.oct=Math.max(-2,Math.min(2,st.oct+d));$('tx-oct').textContent=st.oct;refresh();}
    });
    $('tx-saved').addEventListener('click',e=>{
      const o=e.target.closest('[data-id]'), del=e.target.closest('[data-del]');
      if(o&&app())app().openSong(o.dataset.id);
      if(del){writeSaved(loadSaved().filter(s=>s.id!==del.dataset.del));syncLibrary();renderSaved();}
    });
    window.addEventListener('resize',()=>{if(st.res&&!$('tx-result').hidden)drawRoll();});

    // Route "transcribe" through the app's navigation; every other destination hides this view.
    const C=app();
    if(C&&C.navTo){
      const nav=C.navTo.bind(C), open=C.openSong&&C.openSong.bind(C), view=$('view-transcribe');
      C.navTo=function(d){
        if(d!=='transcribe'){view.classList.remove('active');return nav(d);}
        this.closeMenu&&this.closeMenu();
        if(this.playing)this.stop(true);
        document.querySelectorAll('.view.active').forEach(v=>v.classList.remove('active'));
        view.classList.add('active'); renderSaved();
      };
      if(open)C.openSong=function(id,f){view.classList.remove('active');return open(id,f);};
    }
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
