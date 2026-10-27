const token = document.querySelector('meta[name="splanner-token"]').content;
const $ = (id) => document.getElementById(id);
const state = { view: "today", date: new Date(), events: [], tasks: [], notes: [],
  reminders: [], stickies: [], currentNote: null, editing: null, noteTimer: null,
  noteQueue: Promise.resolve(), noteDirty: false, noteChange: 0, flushNote: null,
  dueShown: new Set(), pollSeconds: 20 };
const names = {today:"Today",calendar:"Calendar",tasks:"My tasks",notes:"Notes",board:"Sticky board"};
const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

function element(tag, className = "", value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}
function button(label, cls, onClick) {
  const b = element("button",cls,label); b.type = "button"; b.addEventListener("click",onClick); return b;
}
async function api(route, method = "GET", body) {
  const res = await fetch(`/api/${route}`, {method,
    headers:{"X-SPlanner-Token":token,...(body?{"Content-Type":"application/json"}:{})},
    ...(body?{body:JSON.stringify(body)}:{})});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
function error(err) { toast(err instanceof Error?err.message:String(err)); }
function toast(message, action) {
  const item = element("div","toast",message);
  if(action) item.append(button(action.label,"text-action",()=>{action.run();item.remove();}));
  $("toasts").prepend(item);
  setTimeout(()=>item.remove(),9000);
}
function localDay(date) { return new Date(date.getFullYear(),date.getMonth(),date.getDate(),12); }
function monday(date) { const d=localDay(date); d.setDate(d.getDate()-(d.getDay()+6)%7); return d; }
function plusDays(date,n) { const d=localDay(date); d.setDate(d.getDate()+n); return d; }
function sameDay(a,b) { return a.toDateString()===b.toDateString(); }
function dateKey(date) { const p=n=>String(n).padStart(2,"0");return `${date.getFullYear()}-${p(date.getMonth()+1)}-${p(date.getDate())}`; }
function localInput(date) { const p=n=>String(n).padStart(2,"0");return `${dateKey(date)}T${p(date.getHours())}:${p(date.getMinutes())}`; }
function labelDate(date,options={}) { return new Intl.DateTimeFormat(undefined,options).format(date); }
function weekDays() { const start=monday(state.date);return Array.from({length:7},(_,i)=>plusDays(start,i)); }
function weekText() { const d=weekDays(); return `${labelDate(d[0],{month:"short",day:"numeric"})} – ${labelDate(d[6],{month:"short",day:"numeric",year:"numeric"})}`; }
function formatTime(s) { return labelDate(new Date(s),{hour:"numeric",minute:"2-digit"}); }
function titleOf(kind) {return ({events:"Event",tasks:"Task",notes:"Note",reminders:"Reminder",stickies:"Sticky note"})[kind]||kind;}

function miniCalendar() {
  const grid=$("mini-calendar"); grid.replaceChildren();
  const date=state.date;
  $("mini-month").textContent=labelDate(date,{month:"long",year:"numeric"});
  days.forEach(d=>grid.append(element("span","mini-cell head",d[0])));
  const first=new Date(date.getFullYear(),date.getMonth(),1,12);
  const start=monday(first);
  for(let i=0;i<42;i++){
    const day=plusDays(start,i); const b=button(day.getDate(),"mini-cell",()=>{state.date=day;changeView("calendar");});
    if(day.getMonth()!==date.getMonth())b.classList.add("other");
    if(sameDay(day,new Date()))b.classList.add("current");
    b.setAttribute("aria-label",labelDate(day,{weekday:"long",month:"long",day:"numeric",year:"numeric"}));
    grid.append(b);
  }
}

function render() {
  $("crumb").textContent=names[state.view];
  $("eyebrow").textContent=state.view==="today"?"A FRESH DAY TO MAKE PROGRESS":"YOUR WORKSPACE";
  const headings={today:"Good to see you.",calendar:"Your calendar",tasks:"Stay on top of things",notes:"Thoughts, in one place",board:"Your idea space"};
  $("title").textContent=headings[state.view];
  $("subtitle").textContent=state.view==="today"?`${labelDate(new Date(),{weekday:"long",month:"long",day:"numeric",year:"numeric"})} · A little progress adds up.`:
    state.view==="calendar"?weekText():state.view==="notes"?"SLDC-backed notes, stored privately on your device.":
    state.view==="board"?"Move ideas around, without losing their content.":"Make space for the work that matters.";
  document.querySelectorAll("[data-view]").forEach(n=>{
    n.classList.toggle("active",n.dataset.view===state.view);
    if(n.dataset.view===state.view)n.setAttribute("aria-current","page");
    else n.removeAttribute("aria-current");
  });
  miniCalendar();
  const target=$("view"); target.replaceChildren();
  if(state.view==="today") renderToday(target);
  if(state.view==="calendar") target.append(schedulePanel());
  if(state.view==="tasks") renderTasks(target);
  if(state.view==="notes") renderNotes(target);
  if(state.view==="board") renderBoard(target);
  $("notification-count").textContent=state.reminders.filter(r=>r.status==="pending"&&new Date(r.due_at)<=new Date()).length;
  $("notification-count").hidden=$("notification-count").textContent==="0";
}
async function flushNoteIfNeeded(){
  if(state.noteDirty&&state.flushNote){await state.flushNote();}
}
async function changeView(view){
  try { await flushNoteIfNeeded(); }
  catch(e){error(e);return;}
  state.view=view;state.currentNote=null;state.flushNote=null;render();
}

function panel(title, trailing){
  const p=element("section","panel panel-pad");
  const head=element("div","panel-title");head.append(element("h2","",title));
  if(trailing)head.append(trailing);p.append(head);return p;
}
function renderToday(target) {
  const stats=element("div","stats");
  const now=new Date();const today=dateKey(now);
  const values=[
    [state.events.filter(e=>dateKey(new Date(e.start_at))===today).length,"Events today"],
    [state.tasks.filter(t=>t.status==="open").length,"Open tasks"],
    [state.notes.length,"Your notes"]];
  values.forEach(([value,label])=>{const c=element("div","stat");c.append(element("strong","",value),element("span","",label));stats.append(c);});
  target.append(stats);
  const grid=element("div","dashboard");grid.append(schedulePanel(true));
  const side=element("div","right-stack");
  const tasks=panel("Your priorities",button("View all  →","text-action",()=>changeView("tasks")));
  appendTasks(tasks,state.tasks.filter(t=>t.status==="open").slice(0,6));
  side.append(tasks);
  const notes=panel("Quick capture",button("New note  ＋","text-action",()=>openEditor("notes")));
  notes.append(element("p","empty",state.notes.length?`${state.notes.length} notes available in your workspace.`:"Ideas don't need to be perfect. Capture the first thought."));
  side.append(notes);
  const rem=panel("Reminders",button("Add alert  ＋","text-action",()=>openEditor("reminders")));
  state.reminders.filter(r=>r.status==="pending").slice(0,4).forEach(r=>{
    const row=element("div","task-item");row.append(element("div","task-copy",r.title+" · "+formatTime(r.due_at)));
    row.append(button("Dismiss","text-action",async()=>{try{await api(`reminders/${r.id}`,"PATCH",{action:"dismiss"});await refresh();}catch(e){error(e);}}));rem.append(row);
  });
  if(!rem.querySelector(".task-item")) rem.append(element("p","empty","All clear. Nothing due right now."));
  side.append(rem);grid.append(side);target.append(grid);
}
function schedulePanel(compact=false) {
  const p=panel(compact?"This week":"Weekly schedule",button("＋ Add event","text-action",()=>openEditor("events")));
  const canvas=element("div","calendar-wrap");
  canvas.append(element("div","day-name"));
  const week=weekDays();
  week.forEach(d=>{const hd=element("div","day-name"+(sameDay(d,new Date())?" is-today":""),labelDate(d,{weekday:"short"}));hd.append(element("strong","",d.getDate()));canvas.append(hd);});
  const hours=element("div","time-sidebar");
  for(let h=8;h<19;h++)hours.append(element("div","time-label",`${String(h).padStart(2,"0")}:00`));
  canvas.append(hours);
  week.forEach(d=>{
    const lane=element("div","day-lane");lane.setAttribute("aria-label",labelDate(d,{dateStyle:"full"}));
    lane.addEventListener("click",ev=>{
      if(ev.target!==lane)return;
      const rect=lane.getBoundingClientRect();const hour=Math.max(8,Math.min(18,8+Math.floor((ev.clientY-rect.top)/64)));
      const start=new Date(d.getFullYear(),d.getMonth(),d.getDate(),hour);
      openEditor("events",null,start);
    });
    state.events.forEach(event=>{
      const start=new Date(event.start_at),end=new Date(event.end_at);
      const dayStart=new Date(d.getFullYear(),d.getMonth(),d.getDate(),8);
      const dayEnd=new Date(d.getFullYear(),d.getMonth(),d.getDate(),19);
      if(end<=dayStart||start>=dayEnd)return;
      const min=Math.max(start.getTime(),dayStart.getTime());
      const max=Math.min(end.getTime(),dayEnd.getTime());
      const top=(min-dayStart.getTime())/3600000*64;
      const height=Math.max(28,(max-min)/3600000*64-2);
      const b=button(event.title,"event-pill color-"+event.color,()=>openEditor("events",event));
      b.style.top=top+"px";b.style.height=height+"px";
      b.append(element("small","",`${formatTime(event.start_at)} – ${formatTime(event.end_at)}`));
      b.title=event.description||event.title;lane.append(b);
    });
    canvas.append(lane);
  });p.append(canvas);return p;
}
function appendTasks(container,tasks){
  if(!tasks.length){container.append(element("p","empty","A clear list. Create a task to get started."));return;}
  tasks.forEach(t=>{
    const row=element("div","task-item"+(t.status==="done"?" done":""));
    const checkbox=element("input");checkbox.type="checkbox";checkbox.checked=t.status==="done";
    checkbox.setAttribute("aria-label",`Mark ${t.title} as ${t.status==="done"?"open":"done"}`);
    checkbox.addEventListener("change",async()=>{try{await api(`tasks/${t.id}`,"PATCH",{status:checkbox.checked?"done":"open",version:t.version});await refresh();}catch(e){error(e);checkbox.checked=!checkbox.checked;}});
    const copy=element("div","task-copy");copy.append(element("div","task-title",t.title));
    copy.append(element("div","task-meta",t.due_at?`Due ${labelDate(new Date(t.due_at),{month:"short",day:"numeric"})}`:"No deadline"));
    row.append(checkbox,copy,element("span","priority "+(t.priority==="high"?"high":""),t.priority));
    row.append(button("⋯","plain",()=>openEditor("tasks",t)));container.append(row);
  });
}
function renderTasks(target){
  const p=panel("All tasks",button("＋ New task","text-action",()=>openEditor("tasks")));
  appendTasks(p,state.tasks.filter(t=>t.status==="open"));
  const completed=state.tasks.filter(t=>t.status==="done");
  if(completed.length){p.append(element("h3","",`Completed · ${completed.length}`));appendTasks(p,completed);}
  target.append(p);
}
function renderNotes(target){
  if(state.currentNote){renderNoteEditor(target);return;}
  const head=element("div","board-actions");head.append(element("p","",`${state.notes.length} saved .sldc documents`),button("＋ New note","primary-button",()=>openEditor("notes")));target.append(head);
  const grid=element("div","notes-grid");
  state.notes.forEach(note=>{const c=element("button","note-card");c.type="button";
    c.append(element("h3","",note.title),element("p","","Open this note to continue writing."),element("small","",`SLDC · Updated ${labelDate(new Date(note.updated_at),{month:"short",day:"numeric"})}`));
    c.addEventListener("click",()=>openNote(note.id));grid.append(c);});
  if(!state.notes.length)grid.append(element("p","empty","Create your first note. It'll be saved as .sldc automatically."));
  target.append(grid);
}
async function openNote(id){
  try{await flushNoteIfNeeded();state.currentNote=await api(`notes/${id}`);state.view="notes";render();}catch(e){error(e);}
}
function renderNoteEditor(target){
  const note=state.currentNote;const p=element("div","panel note-editor");
  const top=element("div","note-toolbar");top.append(button("← All notes","text-action",()=>changeView("notes")));
  const status=element("span","note-save-state","Saved as .sldc");top.append(status);p.append(top);
  const name=element("input");name.value=note.title;name.maxLength=180;name.setAttribute("aria-label","Note title");
  const content=element("textarea");content.value=note.content;content.maxLength=50000;content.setAttribute("aria-label","Note text");
  p.append(name,content);
  async function save(){
    clearTimeout(state.noteTimer);
    if(!state.noteDirty) return;
    const latest={title:name.value,content:content.value};
    const versionOfInput=state.noteChange;status.textContent="Saving…";
    state.noteQueue=state.noteQueue.then(async()=>{
      const saved=await api(`notes/${note.id}`,"PATCH",{...latest,version:state.currentNote.version});
      state.currentNote=saved;
      if(versionOfInput===state.noteChange){state.noteDirty=false;status.textContent="Saved as .sldc";}
      else status.textContent="New changes pending…";
      const i=state.notes.findIndex(n=>n.id===saved.id);if(i>=0)state.notes[i]=saved;
    });
    try{await state.noteQueue;}catch(e){status.textContent="Not saved — recover by retrying";state.noteQueue=Promise.resolve();throw e;}
  }
  state.flushNote=save;
  [name,content].forEach(n=>n.addEventListener("input",()=>{state.noteChange++;state.noteDirty=true;status.textContent="Unsaved changes";clearTimeout(state.noteTimer);state.noteTimer=setTimeout(()=>save().catch(error),850);}));
  const actions=element("div","note-toolbar");actions.append(element("span","note-save-state","Basic text editor. Advanced annotations open in SReader when connected."));
  actions.append(button("Save note","primary-button",()=>save().catch(error)));p.append(actions);target.append(p);
}
async function renderBoard(target){
  const head=element("div","board-actions");head.append(element("p","","Drag notes to arrange your ideas. Click a note to edit its SLDC content."),button("＋ Add sticky","primary-button",()=>openEditor("stickies")));
  target.append(head);const board=element("div","board");board.setAttribute("aria-label","Sticky note board");target.append(board);
  for(const sticky of state.stickies){
    const card=element("div","sticky color-"+sticky.color);card.style.left=sticky.x+"px";card.style.top=sticky.y+"px";
    card.tabIndex=0;card.setAttribute("role","button");card.setAttribute("aria-label","Open sticky note");
    card.append(element("h3","","Loading…"));board.append(card);
    api(`notes/${sticky.note_id}`).then(note=>{card.querySelector("h3").textContent=note.title;
      card.append(element("small","",note.content.slice(0,120)));}).catch(error);
    let dragging=null;let moved=false;
    card.addEventListener("pointerdown",ev=>{
      if(ev.button!==0)return;dragging={x:ev.clientX,y:ev.clientY,l:sticky.x,t:sticky.y};moved=false;card.setPointerCapture(ev.pointerId);
    });
    card.addEventListener("pointermove",ev=>{
      if(!dragging)return;const dx=ev.clientX-dragging.x,dy=ev.clientY-dragging.y;
      if(Math.abs(dx)+Math.abs(dy)>5)moved=true;
      card.style.left=Math.max(0,dragging.l+dx)+"px";card.style.top=Math.max(0,dragging.t+dy)+"px";
    });
    card.addEventListener("pointerup",async ev=>{
      if(!dragging)return;const move=dragging;dragging=null;
      if(moved){try{const updated=await api(`stickies/${sticky.id}`,"PATCH",{x:Math.max(0,move.l+ev.clientX-move.x),y:Math.max(0,move.t+ev.clientY-move.y),version:sticky.version});Object.assign(sticky,updated);}catch(e){error(e);card.style.left=sticky.x+"px";card.style.top=sticky.y+"px";}}
      else await openNote(sticky.note_id);
    });
    card.addEventListener("pointercancel",()=>{dragging=null;card.style.left=sticky.x+"px";card.style.top=sticky.y+"px";});
    card.addEventListener("keydown",ev=>{if(ev.key==="Enter")openNote(sticky.note_id);});
  }
}

function updateModalFields() {
  const kind=$("edit-kind").value,existing=Boolean(state.editing);
  $("dialog-title").textContent=(existing?"Edit ":"New ")+titleOf(kind).toLowerCase();
  $("edit-time-fields").hidden=!(["events","tasks","reminders"].includes(kind));
  $("edit-end-field").hidden=kind!=="events";
  $("edit-start-label").textContent=kind==="events"?"Starts":"Due";
  $("edit-task-fields").hidden=kind!=="tasks";
  $("edit-color-field").hidden=!(["events","stickies"].includes(kind));
  $("edit-content-field").hidden=!(["notes","stickies"].includes(kind));
  $("delete").hidden=!existing||!(kind==="events"||kind==="tasks");
}
function openEditor(kind="events",existing=null,at=new Date()){
  state.editing=existing;
  $("edit-form").reset();$("edit-kind").value=kind;$("edit-kind").disabled=Boolean(existing);
  const rounded=new Date(at.getFullYear(),at.getMonth(),at.getDate(),Math.max(8,at.getHours()),0);
  $("edit-title").value=existing?.title||"";
  const priorStart=existing?.start_at||existing?.due_at;
  $("edit-start").value=(kind==="tasks"&&!priorStart)?"":localInput(priorStart?new Date(priorStart):rounded);
  $("edit-end").value=localInput(existing?.end_at?new Date(existing.end_at):new Date(rounded.getTime()+3600000));
  $("edit-priority").value=existing?.priority||"normal";
  $("edit-color").value=existing?.color||(kind==="stickies"?"amber":"blue");
  $("edit-content").value="";updateModalFields();$("editor").showModal();$("edit-title").focus();
}
async function saveEditor(ev){
  ev.preventDefault();const kind=$("edit-kind").value, title=$("edit-title").value.trim();
  if(!title){$("edit-title").focus();return;}
  let body={title};
  try{
    if(kind==="events"){
      body={...body,start_at:new Date($("edit-start").value).toISOString(),end_at:new Date($("edit-end").value).toISOString(),
        time_zone:Intl.DateTimeFormat().resolvedOptions().timeZone||"UTC",color:$("edit-color").value};
    }else if(kind==="tasks"){
      body.priority=$("edit-priority").value;
      if($("edit-start").value)body.due_at=new Date($("edit-start").value).toISOString();
    }else if(kind==="reminders")body.due_at=new Date($("edit-start").value).toISOString();
    else if(kind==="notes"||kind==="stickies")body.content=$("edit-content").value;
    if(kind==="stickies")body.color=$("edit-color").value;
    if(state.editing){body.version=state.editing.version;await api(`${kind}/${state.editing.id}`,"PATCH",body);}
    else await api(kind,"POST",body);
    $("editor").close();state.editing=null;await refresh();toast(`${titleOf(kind)} saved`);
  }catch(e){error(e);}
}
async function deleteEditing(){
  if(!state.editing)return;
  if(!confirm("Delete this item? This cannot be undone."))return;
  try{const kind=$("edit-kind").value;await api(`${kind}/${state.editing.id}`,"DELETE");$("editor").close();state.editing=null;await refresh();}catch(e){error(e);}
}
async function refresh(){
  const loaded=await api("bootstrap");
  Object.assign(state,{events:loaded.events,tasks:loaded.tasks,notes:loaded.notes,reminders:loaded.reminders,stickies:loaded.stickies,pollSeconds:loaded.reminderPollSeconds});
  render();
}
async function pollReminders(){
  try{
    const due=await api("reminders/due");
    for(const r of due){
      if(state.dueShown.has(r.id))continue;
      state.dueShown.add(r.id);
      toast(`Reminder: ${r.title}`,{label:"Dismiss",run:async()=>{try{await api(`reminders/${r.id}`,"PATCH",{action:"dismiss"});await refresh();}catch(e){error(e);}}});
      if("Notification" in window&&Notification.permission==="granted")new Notification("SPlanner reminder",{body:r.title});
    }
    if(due.length) {const count=$("notification-count");count.hidden=false;count.textContent=due.length;}
  }catch(e){error(e);}
}
function initialize(){
  $("nav").addEventListener("click",ev=>{const target=ev.target.closest("[data-view]");if(target)changeView(target.dataset.view);});
  document.querySelectorAll("[data-nav]").forEach(n=>n.addEventListener("click",()=>{state.date=plusDays(state.date,n.dataset.nav==="next"?7:-7);render();}));
  $("previous").addEventListener("click",()=>{state.date=plusDays(state.date,-7);render();});
  $("next").addEventListener("click",()=>{state.date=plusDays(state.date,7);render();});
  $("today").addEventListener("click",()=>{state.date=new Date();render();});
  $("new").addEventListener("click",()=>openEditor(({tasks:"tasks",notes:"notes",board:"stickies"})[state.view]||"events"));
  $("edit-kind").addEventListener("change",updateModalFields);
  $("edit-form").addEventListener("submit",saveEditor);
  $("delete").addEventListener("click",deleteEditing);
  ["cancel","dialog-close"].forEach(id=>$(id).addEventListener("click",()=>$("editor").close()));
  $("notification-button").addEventListener("click",()=>{
    pollReminders();
    if("Notification" in window&&Notification.permission==="default")Notification.requestPermission();
  });
  const theme=localStorage.getItem("splanner-theme");if(theme==="dark")document.documentElement.dataset.theme="dark";
  $("theme").addEventListener("click",()=>{const dark=document.documentElement.dataset.theme!=="dark";document.documentElement.dataset.theme=dark?"dark":"light";localStorage.setItem("splanner-theme",dark?"dark":"light");});
  window.addEventListener("beforeunload",event=>{if(state.noteDirty){event.preventDefault();event.returnValue="";}});
  refresh().then(()=>{pollReminders();setInterval(pollReminders,Math.max(5,state.pollSeconds)*1000);}).catch(error);
}
initialize();
