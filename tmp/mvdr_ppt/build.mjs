import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire('C:/Users/atr/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/package.json');
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root = process.cwd(), tmp = path.join(root,'tmp/mvdr_ppt'), out = path.join(root,'output/ppt');
await fs.mkdir(tmp,{recursive:true}); await fs.mkdir(out,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
function shape(slide,text,x,y,w,h,size=27,opts={}) {const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}}); s.text=text; s.text.style={fontSize:size,typeface:'Microsoft YaHei',color:opts.color||'#24344A',bold:!!opts.bold,alignment:opts.align||'left',verticalAlignment:'middle',wrap:'square',autoFit:'none',insets:0}; return s;}
function slide(title,n){const s=deck.slides.add();s.background.fill='#FFFFFF';shape(s,title,64,32,1120,52,34,{bold:true,color:'#18334B'});shape(s,'MVDR 接收波束最优性证明',64,676,900,20,14,{color:'#697786'});shape(s,String(n),1180,676,40,20,14,{color:'#697786',align:'right'});return s;}
function t(s,text,y,h=42){shape(s,text,72,y,1136,h,24)}
function eq(s,text,y,h=54){shape(s,text,95,y,1090,h,27,{color:'#153E59',align:'center'})}
{
 const s=slide('定理：存在采用 MVDR 接收波束的最优方案',1);
 t(s,'设 P0 满足原系统模型，且存在全局最优解。',110);
 t(s,'假设 σ² > 0，pⱼ(t) ≥ 0，ω₂ ≥ 0，接收导向向量非零。',165);
 t(s,'感知信道为秩一形式，发射波束允许向零缩放，且无其他波束约束。',220);
 t(s,'则至少存在一个 P0 全局最优解，对每个 UAV 和每个时隙均采用',285);
 eq(s,'gᵢᴹ(t) = Rᵢ(t)⁻¹ aᵣ(θᵢ(t)) / ‖Rᵢ(t)⁻¹ aᵣ(θᵢ(t))‖',345,70);
 eq(s,'Rᵢ(t) = ∑ⱼ ηᵢ,ⱼ(t) pⱼ(t) h꜀ⱼ,ᵤᵢ(t)ᴴ h꜀ⱼ,ᵤᵢ(t) + σ² I',435,70);
 t(s,'其中 gᵢᴹ(t) 满足原问题的单位范数约束；各波束均为复向量。',535);
}
{
 const s=slide('证明（一）：MVDR 接收方向最大化感知 SINR',2);
 t(s,'任取 UAV i 和时隙 t，暂略下标。由原感知信道的秩一结构，有',100);
 eq(s,'A = α aᵣ aₜᴴ，    R = ∑ⱼ ηⱼ pⱼ hⱼᴴ hⱼ + σ² I ≻ 0',157);
 eq(s,'SINR(g, w) = |α|² |aₜᴴ w|² · |gᴴ aᵣ|² / (gᴴ R g)',228);
 t(s,'对 R¹ᐟ²g 与 R⁻¹ᐟ²aᵣ 使用 Cauchy–Schwarz 不等式，得到',300);
 eq(s,'|gᴴ aᵣ|² ≤ (gᴴ R g) (aᵣᴴ R⁻¹ aᵣ)',360);
 eq(s,'|gᴴ aᵣ|² / (gᴴ R g) ≤ aᵣᴴ R⁻¹ aᵣ',425);
 t(s,'当 g 与 R⁻¹aᵣ 同方向时取等；单位范数解即为定理中的 gᴹ。',510);
 t(s,'若发射目标增益为零，所有接收方向的 SINR 均为零。',566);
}
{
 const s=slide('证明（二）：从任意可行解构造新方案',3);
 t(s,'任取 P0 的可行解 ζ。保持位置、匹配、CU 功率及其他变量不变。',105);
 t(s,'先将接收波束替换为 gᴹ，保持发射波束 w 不变。根据上一页，',165);
 eq(s,'s = SINR(g, w)，    sᴹ = SINR(gᴹ, w) ≥ s ≥ 0',225);
 t(s,'进一步令 w′ = c w，其中缩放系数取为',300);
 eq(s,'c = √(s / sᴹ)  （sᴹ > 0）；     c = 0  （sᴹ = 0）',355);
 t(s,'由于 0 ≤ c ≤ 1，新方案满足',425);
 eq(s,'SINR(gᴹ, w′) = c² sᴹ = s，     ‖w′‖² = c²‖w‖² ≤ ‖w‖²',485);
 t(s,'对所有 UAV 和时隙分别执行该构造，所得方案记为 ζ′。',570);
}
{
 const s=slide('证明（三）：感知卸载与计算约束',4);
 t(s,'由新旧感知 SINR 相等，感知信息速率和数据量保持不变：',105);
 eq(s,'Rᵢˢᵉⁿ′(t) = Rᵢˢᵉⁿ(t)，     D̄ˢᵉⁿ Rᵢˢᵉⁿ′(t) = D̄ˢᵉⁿ Rᵢˢᵉⁿ(t)',165);
 t(s,'其余变量保持不变，所以原可行解满足的两项约束仍然成立：',240);
 eq(s,'Dᵤᵢᵒᶠᶠ(t) Rᵢᵒᶠᶠ(t) ≥ D̄ˢᵉⁿ Rᵢˢᵉⁿ′(t)',300);
 eq(s,'D̄ˢᵉⁿ + Dᵤᵢᵒᶠᶠ(t) + Cˢᵉⁿ D̄ˢᵉⁿ Rᵢˢᵉⁿ′(t) / fᵤᵢ(t) ≤ Dₘₐₓˢᵉⁿ',375);
 t(s,'BS 的感知计算能耗不变，感知 SINR 阈值仍成立。',460);
 t(s,'又因 ‖wᵢ′(t)‖² ≤ ‖wᵢ(t)‖²，感知发射功率上限仍成立。',520);
}
{
 const s=slide('证明（四）：CU 卸载与其余约束',5);
 t(s,'感知期间，对共享频谱 CU 的干扰满足',105);
 eq(s,'|hᵤᵢ,ᴮˢ(t)ᴴ wᵢ′(t)|² = cᵢ(t)² |hᵤᵢ,ᴮˢ(t)ᴴ wᵢ(t)|²',165);
 eq(s,'≤ |hᵤᵢ,ᴮˢ(t)ᴴ wᵢ(t)|²',225);
 t(s,'因此第一阶段 CU 速率满足 Rᵢ,ⱼ⁽¹⁾′(t) ≥ Rᵢ,ⱼ⁽¹⁾(t)。',295);
 t(s,'第二、第三阶段速率及各阶段时长不变，CU 成功卸载约束仍成立。',355);
 t(s,'轨迹、匹配、CU 功率、卸载波束、时长及计算频率保持原值。',430);
 t(s,'因此其余约束，包括跨时隙的轨迹约束，均保持成立。',490);
 t(s,'综上，ζ′ 是 P0 的可行解。',560);
}
{
 const s=slide('证明（五）：比较新旧方案的目标值',6);
 t(s,'感知数据量、计算频率和其余变量不变，只有感知发射能耗可能改变。',105);
 t(s,'令 E(ζ) 为 P0 的长期平均加权能耗。由 ω₂ ≥ 0，可得',170);
 eq(s,'E(ζ′) − E(ζ)',235);
 eq(s,'= − (ω₂ D̄ˢᵉⁿ / tₘₐₓ) ∑ₜ ∑ᵢ [1 − cᵢ(t)²] ‖wᵢ(t)‖²',300);
 eq(s,'≤ 0',365);
 t(s,'因此，任意 P0 可行解都能构造出一个接收波束为 gᴹ 的可行解，',440);
 t(s,'并且构造解的目标值不大于原方案的目标值。',500);
}
{
 const s=slide('证明（六）：最优值相等',7);
 t(s,'令 ζ★ 为 P0 的一个全局最优解，其目标值记为 E★。',105);
 t(s,'对 ζ★ 应用上述构造，得到采用 MVDR 接收波束的可行解 ζᴹ。',170);
 t(s,'根据目标值比较，有',235);
 eq(s,'E(ζᴹ) ≤ E(ζ★) = E★',290);
 t(s,'又因为 ζᴹ 对 P0 可行，且 E★ 是 P0 的全局最小值，有',355);
 eq(s,'E(ζᴹ) ≥ E★',410);
 eq(s,'E(ζᴹ) = E★',490);
 t(s,'故 ζᴹ 是采用 MVDR 接收波束的 P0 全局最优解。证毕。',565);
}
for(let i=0;i<deck.slides.items.length;i++){const b=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
const draft=path.join(tmp,'draft.pptx'); await (await PresentationFile.exportPptx(deck)).save(draft);
const skill='C:/Users/atr/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:path.join(out,'MVDR_证明_中文.pptx'),pythonExecutable:'C:/Users/atr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
console.log(JSON.stringify(result));
