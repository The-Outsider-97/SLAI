import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

document.addEventListener('DOMContentLoaded', async () => {
  const apiKey = localStorage.getItem('mindweave_llm_api_key');
  const responsesPath = '/mindweave/templates/responses.json';
  const apiStatusEl = document.getElementById('api-status');
  const chatInput = document.getElementById('player-input');
  const sendBtn = document.getElementById('btn-send');
  const chatHistory = document.getElementById('chat-history');

  const phaseTitle = document.getElementById('phase-title');
  const phaseContent = document.getElementById('phase-content');
  const phaseButtons = Array.from(document.querySelectorAll('.phase-btn'));
  const telemetryEl = document.getElementById('telemetry');
  const finalScoreEl = document.getElementById('final-score');
  const protocolScoreListEl = document.getElementById('protocol-score-list');
  const objectiveLog = document.getElementById('objective-log');
  const btnHome = document.getElementById('btn-home');
  const btnSettings = document.getElementById('btn-settings');
  const btnNewGame = document.getElementById('btn-new-game');
  const btnAcademy = document.getElementById('btn-academy');
  const btnIqTest = document.getElementById('btn-iq-test');
  const btnEqTest = document.getElementById('btn-eq-test');
  const settingsModal = document.getElementById('settings-modal');
  const settingsClose = document.getElementById('settings-close');
  const bgmVolumeInput = document.getElementById('bgm-volume');
  const sfxVolumeInput = document.getElementById('sfx-volume');
  const bgmMuteInput = document.getElementById('bgm-mute');
  const sfxMuteInput = document.getElementById('sfx-mute');
  const voiceVolumeInput = document.getElementById('voice-volume');
  const voiceMuteInput = document.getElementById('voice-mute');
  const endgameModal = document.getElementById('endgame-modal');
  const endgameClose = document.getElementById('endgame-close');
  const endgameRestart = document.getElementById('endgame-restart');
  const endgameFinalScore = document.getElementById('endgame-final-score');
  const endgameSummary = document.getElementById('endgame-summary');

  const iqValue = document.getElementById('iq-value');
  const iqBar = document.getElementById('iq-bar');
  const eqSyncValue = document.getElementById('eq-sync-value');
  const eqSyncBar = document.getElementById('eq-sync-bar');
  const progressValue = document.getElementById('progress-value');
  const progressBar = document.getElementById('progress-bar');
  const emotionIndicator = document.getElementById('emotion-indicator');

  const audioLibrary = {
    bgm: ['../src/audio/bg_01.mp3', '../src/audio/bg_02.mp3', '../src/audio/bg_03.mp3', '../src/audio/bg_04.mp3'],
    sfx: {
      error: '../src/audio/error.mp3',
      correct: '../src/audio/correct.mp3',
      wrong: '../src/audio/wrong.mp3',
      heartbeat: '/src/audio/heartbeat.m4a',
      ping: '/src/audio/ping.mp3',
      powerDown: '/src/audio/power_down.m4a',
      achieve: '/src/audio/achieve.mp3',
    },
    voice: {
      briefing: '../src/audio/A7_01a_mission_briefing.m4a',
      briefingAcknowledge: '../src/audio/A7_02_briefing_acknowledge.m4a',
      protocol1: '../src/audio/A7_campaign_protocol_01a.m4a',
      protocol2: '../src/audio/A7_campaign_protocol_02a.m4a',
      protocol3: '../src/audio/A7_campaign_protocol_03a.m4a',
      protocol4: '../src/audio/A7_campaign_protocol_04a.m4a',
      chatError: '../src/audio/A7_chat_error.m4a',
      calmResponse: '../src/audio/A7_calm_response_a.m4a',
      stressResponse: '../src/audio/A7_stress_response_a.m4a',
      thinkingResponse: '../src/audio/A7_thinking_response_a.m4a',
      ambiguousResponse: '../src/audio/A7_chat_error_a.mp4',
      debriefReceived: '../src/audio/A7_final_debrief.m4a',
    }
  };

  const audioState = {
    bgmVolume: 0.25,
    sfxVolume: 0.75,
    voiceVolume: 1,
    bgmMuted: false,
    sfxMuted: false,
    voiceMuted: false,
    bgmCurrent: null,
    bgmNext: null,
    bgmFadeInterval: null,
    bgmTrackIndex: 0,
  };

  const voiceCache = new Map();
  const sfxCache = new Map();

  const voicePriority = {
    protocol: 4,
    sequence: 3,
    reply: 2,
    reaction: 1,
  };

  const voiceQueue = [];
  let activeVoiceJob = null;

  let hasPlayedAchieveCue = false;

  const aiCoachState = {
    lastHintAt: 0,
  };

  const gameState = {
    phase: 'briefing',
    iqScore: 45,
    eqScore: 80,
    progress: 0,
    completedObjectives: new Set(),
    activeSessionId: null,
    nbackSequence: [],
    resourceTarget: Math.floor(Math.random() * 13) + 8,
    threatCodePuzzle: null,
    regulationBreaths: 0,
    pulseSequence: {
      targetBpm: null,
      tolerance: 0.05,
      tapCount: 0,
      lastTapTime: 0,
      tapIntervals: [],
      awaitingInput: false,
      active: false,
    },
    challengeAttempts: {
      nback_clear: { count: 0, locked: false },
      resource_clear: { count: 0, locked: false },
      logic_clear: { count: 0, locked: false },
      iq_switch_clear: { count: 0, locked: false },
      iq_planning_clear: { count: 0, locked: false },
      micro_clear: { count: 0, locked: false },
      eq_reframe_clear: { count: 0, locked: false },
      eq_boundary_clear: { count: 0, locked: false },
    },
    protocolScore: {
      iq: 0,
      eq: 0,
      debrief: 0,
    },
    scoring: {
      errors: 0,
      lockedProtocols: 0,
      chatSupports: 0,
      completedRun: false,
    },
  };

  const protocolWeights = {
    iq: 50,
    eq: 35,
    debrief: 15,
  };

  const orderedPhases = ['briefing', 'iq', 'eq', 'debrief'];

  const objectiveDefinitions = [
    ['briefing_read', 'Complete mission briefing'],
    ['briefing_plan', 'Lock in stabilization plan'],
    ['nback_clear', 'Pass dual n-back memory drill'],
    ['resource_clear', 'Stabilize resource routing'],
    ['logic_clear', 'Repair logic gate'],
    ['iq_switch_clear', 'Complete task-switching relay'],
    ['iq_planning_clear', 'Solve planning optimization drill'],
    ['iq_threatcode_clear', 'Resolve anomaly threat code'],
    ['micro_clear', 'Identify micro-expression'],
    ['eq_reframe_clear', 'Apply cognitive reappraisal'],
    ['eq_boundary_clear', 'Choose co-regulation boundary response'],
    ['eq_bridge_clear', 'Deliver active listening response'],
    ['regulation_clear', 'Reproduce Architect rhythm'],
    ['debrief_submit', 'Submit metacognitive debrief'],
    ['debrief_commitment', 'Define two transfer commitments'],
  ];

  apiStatusEl.textContent = apiKey ? 'Uplink Secure. AI Backend + Key Active.' : 'Uplink Secure. AI Backend Active (no external key).';
  apiStatusEl.classList.replace('text-yellow-400', 'text-green-400');
  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatInput.focus();

  btnHome.addEventListener('click', () => {
    window.location.href = '/index.html';
  });

  btnSettings.addEventListener('click', () => {
    openSettings();
  });

  btnNewGame.addEventListener('click', () => {
    resetGameState();
  });

  btnAcademy.addEventListener('click', () => {
    window.location.href = '/mindweave/pages/academy.html';
  });

  btnIqTest.addEventListener('click', () => {
    if (btnIqTest.disabled) return;
    window.location.href = '/mindweave/pages/iq.html';
  });

  btnEqTest.addEventListener('click', () => {
    if (btnEqTest.disabled) return;
    window.location.href = '/mindweave/pages/eq.html';
  });

  document.getElementById('btn-exit').addEventListener('click', () => {
    window.location.href = '/index.html';
  });

  const container = document.getElementById('canvas-container');
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x020617, 0.04);
  const aspect = window.innerWidth / window.innerHeight;
  const frustumSize = 20;
  const camera = new THREE.OrthographicCamera(frustumSize * aspect / -2, frustumSize * aspect / 2, frustumSize / 2, frustumSize / -2, 1, 1000);
  camera.position.set(20, 20, 20);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 2);
  dirLight.position.set(10, 20, 10);
  scene.add(dirLight);
  const redLight = new THREE.PointLight(0xef4444, 1.5, 20);
  redLight.position.set(-5, 5, -5);
  scene.add(redLight);

  const gridGroup = new THREE.Group();
  const tileGeo = new THREE.BoxGeometry(1.9, 0.5, 1.9);
  for (let i = -3; i < 4; i++) {
    for (let j = -3; j < 4; j++) {
      if (Math.random() > 0.85) continue;
      const mat = new THREE.MeshStandardMaterial({ color: Math.random() > 0.9 ? 0x38bdf8 : 0x1e293b, emissive: 0x0ea5e9, emissiveIntensity: Math.random() > 0.9 ? 0.5 : 0.08 });
      const mesh = new THREE.Mesh(tileGeo, mat);
      mesh.position.set(i * 2, (Math.random() * 0.5) - 0.25, j * 2);
      mesh.userData.baseY = mesh.position.y;
      gridGroup.add(mesh);
    }
  }
  scene.add(gridGroup);

  const npcMesh = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.5, 1),
    new THREE.MeshPhysicalMaterial({ color: 0xffffff, emissive: 0x38bdf8, roughness: 0.1, transmission: 0.8, thickness: 1.0 })
  );
  npcMesh.position.set(0, 3, 0);
  scene.add(npcMesh);

  const emotionColors = { neutral: 0x38bdf8, stress: 0xef4444, calm: 0x10b981, thinking: 0xa855f7 };
  let currentEmotion = 'neutral';

  let responseTemplates = {};

  async function loadResponseTemplates() {
    try {
      const response = await fetch(responsesPath);
      if (!response.ok) throw new Error('Unable to load Architect-7 response templates');
      responseTemplates = await response.json();
    } catch (error) {
      appendChat('SYSTEM', `Response template load failed: ${error.message}`, 'text-yellow-400');
      responseTemplates = {};
    }
  }

  function pickResponse(group, fallbackText, fallbackVoice = null) {
    const options = Array.isArray(responseTemplates[group]) ? responseTemplates[group] : [];
    if (!options.length) return { text: fallbackText, voice: fallbackVoice };
    const choice = options[Math.floor(Math.random() * options.length)] || {};
    return {
      text: choice.text || fallbackText,
      voice: choice.voice || fallbackVoice,
    };
  }

  function resolveAudioPath(path) {
    return path;
  }

  function createAudio(path, loop = false) {
    const audio = new Audio(resolveAudioPath(path));
    audio.loop = loop;
    audio.preload = 'auto';
    return audio;
  }

  function cacheSfx() {
    Object.entries(audioLibrary.sfx).forEach(([key, path]) => {
      sfxCache.set(key, createAudio(path));
    });
  }

  function playSfx(kind) {
    if (audioState.sfxMuted) return;
    const base = sfxCache.get(kind) || createAudio(audioLibrary.sfx[kind]);
    const instance = base.cloneNode();
    instance.volume = audioState.sfxVolume;
    instance.play().catch(() => {});
  }

  function playPulseBeat() {
    if (audioState.sfxMuted) return;
    const pulse = createAudio(audioLibrary.sfx.heartbeat);
    pulse.volume = audioState.sfxVolume;
    pulse.play().catch(() => {});
  }

  async function playArchitectPulseSequence(bpm) {
    const beatIntervalMs = Math.round((60 / bpm) * 1000);
    for (let i = 0; i < 4; i += 1) {
      playPulseBeat();
      appendChat('SYSTEM', `[Architect Pulse] Beat ${i + 1}/4 @ ${bpm} BPM`, 'text-sky-300');
      await new Promise((resolve) => setTimeout(resolve, beatIntervalMs));
    }
  }

  function processVoiceQueue() {
    if (activeVoiceJob || !voiceQueue.length) return;
    const nextJob = voiceQueue.shift();
    const cached = voiceCache.get(nextJob.path) || createAudio(nextJob.path);
    voiceCache.set(nextJob.path, cached);
    const instance = cached.cloneNode();
    activeVoiceJob = { ...nextJob, audio: instance };

    instance.volume = audioState.voiceMuted ? 0 : audioState.voiceVolume;
    instance.play().catch(() => {
      activeVoiceJob = null;
      if (nextJob.resolve) nextJob.resolve();
      processVoiceQueue();
    });

    instance.addEventListener('ended', () => {
      const finish = () => {
        activeVoiceJob = null;
        if (nextJob.resolve) nextJob.resolve();
        processVoiceQueue();
      };
      if (nextJob.delayMs) {
        setTimeout(finish, nextJob.delayMs);
      } else {
        finish();
      }
    }, { once: true });
  }

  function enqueueVoice(path, options = {}) {
    if (!path || audioState.voiceMuted) return Promise.resolve();
    const {
      priority = voicePriority.reaction,
      clearQueue = false,
      interrupt = false,
      delayMs = 0,
    } = options;

    if (clearQueue) voiceQueue.length = 0;

    if (interrupt && activeVoiceJob?.audio) {
      activeVoiceJob.audio.pause();
      activeVoiceJob.audio.currentTime = 0;
      activeVoiceJob = null;
    }

    return new Promise((resolve) => {
      voiceQueue.push({ path, priority, delayMs, resolve });
      voiceQueue.sort((a, b) => b.priority - a.priority);
      processVoiceQueue();
    });
  }

  function fadeTrack(audio, from, to, durationMs, onDone) {
    const steps = 20;
    const delta = (to - from) / steps;
    let i = 0;
    audio.volume = Math.max(0, Math.min(1, from));
    const id = setInterval(() => {
      i += 1;
      audio.volume = Math.max(0, Math.min(1, from + (delta * i)));
      if (i >= steps) {
        clearInterval(id);
        if (onDone) onDone();
      }
    }, Math.max(50, durationMs / steps));
    return id;
  }

  function startBgmTrack(trackIndex = 0) {
    const totalTracks = audioLibrary.bgm.length;
    if (!totalTracks) return;

    const normalizedIndex = ((trackIndex % totalTracks) + totalTracks) % totalTracks;
    audioState.bgmTrackIndex = normalizedIndex;

    if (audioState.bgmCurrent) {
      audioState.bgmCurrent.pause();
      audioState.bgmCurrent.currentTime = 0;
      audioState.bgmCurrent = null;
    }

    const nextTrack = createAudio(audioLibrary.bgm[normalizedIndex], false);
    audioState.bgmCurrent = nextTrack;
    nextTrack.volume = audioState.bgmMuted ? 0 : audioState.bgmVolume;
    nextTrack.addEventListener('ended', () => {
      startBgmTrack((normalizedIndex + 1) % totalTracks);
    }, { once: true });
    nextTrack.play().catch(() => {});
  }

  function startBgm() {
    startBgmTrack(0);
  }

  function applyAudioSettings() {
    if (audioState.bgmCurrent) audioState.bgmCurrent.volume = audioState.bgmMuted ? 0 : audioState.bgmVolume;
    if (activeVoiceJob?.audio) activeVoiceJob.audio.volume = audioState.voiceMuted ? 0 : audioState.voiceVolume;
    if (audioState.bgmNext) audioState.bgmNext.volume = audioState.bgmMuted ? 0 : audioState.bgmVolume;
  }

  function openSettings() {
    settingsModal.classList.remove('hidden');
    settingsModal.setAttribute('aria-hidden', 'false');
  }

  function closeSettings() {
    settingsModal.classList.add('hidden');
    settingsModal.setAttribute('aria-hidden', 'true');
  }

  function renderTelemetry() {
    telemetryEl.innerHTML = `
      <div>Phase: ${gameState.phase.toUpperCase()}</div>
      <div>Objectives: ${gameState.completedObjectives.size}/${objectiveDefinitions.length}</div>
      <div>Session: ${gameState.activeSessionId || 'pending'}</div>
      <div>Breaths: ${gameState.regulationBreaths}</div>
    `;
  }

  function markObjective(id, label) {
    if (!gameState.completedObjectives.has(id)) {
      gameState.completedObjectives.add(id);
      appendChat('SYSTEM', `[Objective complete] ${label}`, 'text-emerald-400');
      const percent = Math.round((gameState.completedObjectives.size / objectiveDefinitions.length) * 100);
      gameState.progress = percent;
      progressValue.textContent = `${percent}%`;
      progressBar.style.width = `${percent}%`;
      renderObjectives();
    }
    renderTelemetry();
  }

  function renderObjectives() {
    objectiveLog.innerHTML = objectiveDefinitions
      .map(([id, label]) => `<div class="${gameState.completedObjectives.has(id) ? 'text-emerald-400' : 'text-slate-500'}">${gameState.completedObjectives.has(id) ? '✓' : '•'} ${label}</div>`)
      .join('');
  }

  function isPhaseComplete(phase) {
    if (phase === 'briefing') return gameState.completedObjectives.has('briefing_read') && gameState.completedObjectives.has('briefing_plan');
    if (phase === 'iq') return gameState.completedObjectives.has('nback_clear') && gameState.completedObjectives.has('resource_clear') && gameState.completedObjectives.has('logic_clear') && gameState.completedObjectives.has('iq_switch_clear') && gameState.completedObjectives.has('iq_planning_clear') && gameState.completedObjectives.has('iq_threatcode_clear');
    if (phase === 'eq') return gameState.completedObjectives.has('micro_clear') && gameState.completedObjectives.has('eq_reframe_clear') && gameState.completedObjectives.has('eq_boundary_clear') && gameState.completedObjectives.has('eq_bridge_clear') && gameState.completedObjectives.has('regulation_clear');
    if (phase === 'debrief') return gameState.completedObjectives.has('debrief_submit') && gameState.completedObjectives.has('debrief_commitment');
    return false;
  }

  function isPhaseAccessible(targetPhase) {
    const targetIndex = orderedPhases.indexOf(targetPhase);
    if (targetIndex <= 0) return true;
    for (let i = 0; i < targetIndex; i += 1) {
      if (!isPhaseComplete(orderedPhases[i])) return false;
    }
    return true;
  }

  function updatePhaseButtonStates() {
    phaseButtons.forEach((btn) => {
      const phase = btn.dataset.phase;
      const unlocked = isPhaseAccessible(phase);
      btn.disabled = !unlocked;
      btn.classList.toggle('opacity-50', !unlocked);
      btn.classList.toggle('cursor-not-allowed', !unlocked);
    });
  }

  function updateAdvancedAccessButtons() {
    const { totalScore } = calculateFinalScore();
    const unlocked = totalScore > 84;
    btnIqTest.disabled = !unlocked;
    btnEqTest.disabled = !unlocked;
    btnIqTest.innerHTML = unlocked ? 'IQ Test <span class="text-emerald-400">[ready]</span>' : 'IQ Test <span class="text-slate-500">[locked]</span>';
    btnEqTest.innerHTML = unlocked ? 'EQ Test <span class="text-emerald-400">[ready]</span>' : 'EQ Test <span class="text-slate-500">[locked]</span>';
  }

  async function blinkAcademyButton(times = 3) {
    for (let i = 0; i < times; i += 1) {
      btnAcademy.classList.add('academy-attention-blink');
      await new Promise((resolve) => setTimeout(resolve, 220));
      btnAcademy.classList.remove('academy-attention-blink');
      if (i < times - 1) await new Promise((resolve) => setTimeout(resolve, 140));
    }
  }

  async function suggestAcademy(templateGroupOrMessage, tone = 'recommend') {
    let message, voice;
    if (typeof templateGroupOrMessage === 'string' && responseTemplates && responseTemplates[templateGroupOrMessage]) {
      const response = pickResponse(templateGroupOrMessage, 'Weaver Academy can help.', null);
      message = response.text;
      voice = response.voice;
    } else {
      message = templateGroupOrMessage;
      voice = null;
    }
    const prefix = tone === 'urgent' ? 'SYSTEM' : 'Architect-7';
    const color = tone === 'urgent' ? 'text-amber-300' : 'text-cyan-200';
    appendChat(prefix, message, color);
    if (voice) {
      enqueueVoice(voice, { priority: voicePriority.sequence });
    }
    await blinkAcademyButton(3);
  }

  function getAiCoachHint(message) {
    const text = message.toLowerCase();
    if (/(stuck|confus|hard|don't get|dont get|help)/.test(text)) {
      return 'Language Agent: Run a 3-step loop — (1) restate the goal, (2) write one rule, (3) test one candidate answer before committing.';
    }
    if (/(anxious|panic|stres|overwhelm|frustrat)/.test(text)) {
      return 'Language Agent: Name the pressure in one sentence, take one breath cycle, then send Architect-7 a calm directive with one concrete next step.';
    }
    if (/(plan|strategy|next|what should i do|guide)/.test(text)) {
      return 'Language Agent: Suggested plan — Scan objective text, eliminate one wrong option, then commit only after a quick self-check.';
    }
    return null;
  }

  function updateNPCState(emotion, textIndicator) {
    currentEmotion = emotion;
    emotionIndicator.textContent = `Analysis: ${textIndicator}`;
    npcMesh.material.emissive.copy(new THREE.Color(emotionColors[emotion] || emotionColors.neutral));
    npcMesh.material.emissiveIntensity = 0.8;
  }

  function updateBars() {
    iqValue.textContent = `${gameState.iqScore}%`;
    iqBar.style.width = `${gameState.iqScore}%`;
    eqSyncValue.textContent = `${gameState.eqScore}%`;
    eqSyncBar.style.width = `${gameState.eqScore}%`;
    eqSyncBar.style.backgroundColor = gameState.eqScore < 40 ? 'var(--traffic-red)' : gameState.eqScore < 70 ? '#eab308' : '#10b981';
  }

  function calculateAttemptMultiplier(attemptCount) {
    if (attemptCount <= 1) return 1;
    if (attemptCount === 2) return 0.55;
    return 0.2;
  }

  function registerChallengeAttempt(challengeId, onSuccess, onFailure, options = {}) {
    const challenge = gameState.challengeAttempts[challengeId];
    if (!challenge) return false;

    if (challenge.locked) {
      appendChat('SYSTEM', 'Protocol already resolved. Additional submissions are ignored.', 'text-slate-400');
      playSfx('error');
      return false;
    }

    if (challenge.count >= 3) {
      gameState.scoring.errors += 1;
      appendChat('SYSTEM', 'Maximum attempts reached for this protocol.', 'text-yellow-400');
      playSfx('error');
      return false;
    }

    challenge.count += 1;
    const attemptCount = challenge.count;

    if (onSuccess(attemptCount)) {
      challenge.locked = true;
      return true;
    }

    gameState.scoring.errors += 1;
    if (attemptCount >= 3) {
      challenge.locked = true;
      gameState.scoring.lockedProtocols += 1;
      appendChat('SYSTEM', 'Protocol locked after 3 attempts.', 'text-yellow-400');
      if (!options.skipAcademyPromptOnLock) {
        suggestAcademy('failed_attempt', 'urgent');
      }
      playSfx('error');
    } else if (onFailure) {
      onFailure(attemptCount);
    }

    return false;
  }

  function awardProtocolScore(protocol, basePoints, attemptCount = 1) {
    const multiplier = calculateAttemptMultiplier(attemptCount);
    const weightedScore = Math.round(basePoints * multiplier);
    gameState.protocolScore[protocol] = Math.min(protocolWeights[protocol], gameState.protocolScore[protocol] + weightedScore);
    renderFinalScore();
  }

  function calculateFinalScore() {
    const rawScore = gameState.protocolScore.iq + gameState.protocolScore.eq + gameState.protocolScore.debrief;
    const unresolved = objectiveDefinitions.length - gameState.completedObjectives.size;
    const penalties = (gameState.scoring.errors * 2) + (gameState.scoring.lockedProtocols * 5) + unresolved;
    return {
      rawScore,
      penalties,
      totalScore: Math.max(0, Math.min(100, rawScore - penalties)),
    };
  }

  function maybeShowEndgameModal() {
    if (gameState.scoring.completedRun || !isPhaseComplete('debrief')) return;
    const score = calculateFinalScore();
    gameState.scoring.completedRun = true;
    endgameFinalScore.textContent = `${score.totalScore} / 100`;
    endgameSummary.innerHTML = [
      `<div>Raw protocol score: ${score.rawScore}/100</div>`,
      `<div>Penalties applied: -${score.penalties}</div>`,
      `<div>Errors logged: ${gameState.scoring.errors}</div>`,
      `<div>Locked protocols: ${gameState.scoring.lockedProtocols}</div>`,
      `<div>Objectives completed: ${gameState.completedObjectives.size}/${objectiveDefinitions.length}</div>`,
    ].join('');
    endgameModal.classList.remove('hidden');
    endgameModal.setAttribute('aria-hidden', 'false');
  }

  function closeEndgameModal() {
    endgameModal.classList.add('hidden');
    endgameModal.setAttribute('aria-hidden', 'true');
  }

  function renderFinalScore() {
    const { totalScore } = calculateFinalScore();
    finalScoreEl.textContent = `${totalScore} / 100`;
    if (totalScore >= 85 && !hasPlayedAchieveCue) {
      hasPlayedAchieveCue = true;
      playSfx('achieve');
    } else if (totalScore < 85) {
      hasPlayedAchieveCue = false;
    }
    updateAdvancedAccessButtons();
    protocolScoreListEl.innerHTML = [
      `IQ Protocol: ${gameState.protocolScore.iq}/${protocolWeights.iq}`,
      `EQ Protocol: ${gameState.protocolScore.eq}/${protocolWeights.eq}`,
      `Debrief Protocol: ${gameState.protocolScore.debrief}/${protocolWeights.debrief}`,
    ].map((line) => `<div>${line}</div>`).join('');
  }

  function appendChat(sender, message, colorClass = 'text-white', voicePath = null) {
    const div = document.createElement('div');
    const senderColor = sender === 'Weaver' ? 'text-[var(--traffic-red)]' : 'text-[var(--neural-blue)]';
    div.innerHTML = `<span class="${senderColor}">> ${sender}:</span> <span class="${colorClass}">${message}</span>`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    if (voicePath) enqueueVoice(voicePath, { priority: voicePriority.reply });
  }

  async function ensureMindweaveSelected() {
    const selectionResp = await fetch('/api/select-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game: 'mindweave' })
    });
    if (!selectionResp.ok) throw new Error('Unable to initialize Mindweave backend');
    const selectedPayload = await selectionResp.json();
    gameState.activeSessionId = selectedPayload.session_id || null;
  }

  async function sendTaskMessage(message, taskType = 'npc_dialogue') {
    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-Mindweave-API-Key': apiKey } : {}) },
        body: JSON.stringify({
          session_id: gameState.activeSessionId,
          player_id: 'weaver',
          message,
          task_type: taskType,
          telemetry: { phase: gameState.phase, iq: gameState.iqScore, eq: gameState.eqScore, progress: gameState.progress, chat_supports: gameState.scoring.chatSupports, objectives_completed: gameState.completedObjectives.size, timestamp: Date.now() }
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || 'Chat request failed');
      updateNPCState(payload.emotion || 'neutral', payload.analysis || 'Synced');
      gameState.eqScore = Math.max(0, Math.min(100, gameState.eqScore + Number(payload.eq_delta || 0)));
      updateBars();
      const reply = payload.reply || 'No response generated.';
      const coachingCue = payload.coaching_hint || payload.agent_hint;
      const normalizedReply = reply.toLowerCase();
      let replyVoice = payload.voice || audioLibrary.voice.calmResponse;
      if (normalizedReply.includes('ambiguous')) replyVoice = payload.voice || audioLibrary.voice.ambiguousResponse;
      if (normalizedReply.includes('stress') || normalizedReply.includes('unstable')) replyVoice = payload.voice || audioLibrary.voice.stressResponse;
      appendChat('Architect-7', reply, 'text-white', replyVoice);
      if (coachingCue) {
        appendChat('Language Agent', coachingCue, 'text-violet-200');
      }
      playSfx('correct');
    } catch (error) {
      appendChat('SYSTEM', `Backend sync failed: ${error.message}`, 'text-red-500');
      const errorResponse = pickResponse('chat_error', 'Input acknowledged. However, the emotional context is ambiguous. Please recalibrate your active listening protocols.', audioLibrary.voice.chatError);
      appendChat('Architect-7', errorResponse.text, 'text-slate-200');
      await enqueueVoice(errorResponse.voice || audioLibrary.voice.chatError, { priority: voicePriority.sequence, clearQueue: true, interrupt: true });
      await enqueueVoice(audioLibrary.voice.ambiguousResponse, { priority: voicePriority.sequence });
      playSfx('error');
      updateNPCState('stress', 'Link Unstable');
    }
    renderTelemetry();
  }

  function generateThreatCodePuzzle() {
    const alpha = Math.floor(Math.random() * 6) + 2;
    const beta = Math.floor(Math.random() * 6) + 2;
    const variants = [
      {
        prompt: `Decode emergency key: if alpha=${alpha} and beta=${beta}, enter alpha² + beta² + alpha.`,
        answer: (alpha ** 2) + (beta ** 2) + alpha,
      },
      {
        prompt: `Decode emergency key: if alpha=${alpha} and beta=${beta}, enter (alpha × beta) + (alpha²).`,
        answer: (alpha * beta) + (alpha ** 2),
      },
      {
        prompt: `Decode emergency key: if alpha=${alpha} and beta=${beta}, enter (alpha² + beta²) − beta.`,
        answer: (alpha ** 2) + (beta ** 2) - beta,
      },
    ];
    return variants[Math.floor(Math.random() * variants.length)];
  }

  function launchProtocolOneFromHome() {
    markObjective('briefing_read', 'Briefing acknowledged from Home launch');
    appendChat('SYSTEM', 'Home launch synchronized. Awaiting briefing confirmation for Protocol 1.', 'text-slate-400');
  }

  async function playStartupSequence() {
    // Clear any ongoing audio
    voiceQueue.length = 0;
    if (activeVoiceJob?.audio) {
      activeVoiceJob.audio.pause();
      activeVoiceJob.audio.currentTime = 0;
      activeVoiceJob = null;
    }

    // 1s pause
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Intro
    const intro = pickResponse('intro', 'Weaver, the socio-cognitive lattice is collapsing.', audioLibrary.voice.briefing);
    appendChat('Architect-7', intro.text, 'text-white');
    await enqueueVoice(intro.voice, { priority: voicePriority.protocol });

    // 0.5s pause
    await new Promise(resolve => setTimeout(resolve, 500));

    // Academy recommendation
    const recommendText = "Before Protocol 1 intensifies, Weaver Academy can give you a fast calibration pass on mission skills.";
    const recommendVoice = "../src/audio/A7_recommend.m4a";
    appendChat('Language Agent', recommendText, 'text-violet-200');
    await enqueueVoice(recommendVoice, { priority: voicePriority.sequence });

    // Blink Academy button
    await blinkAcademyButton(3);

    // 1s pause
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Protocol 1 voice (briefing)
    const briefing = pickResponse('briefing', 'Campaign Protocol 1 engaged.', audioLibrary.voice.protocol1);
    enqueueVoice(briefing.voice, { priority: voicePriority.protocol });

    // Set phase briefing without double voice
    setPhase('briefing', { skipProtocolVoice: true });
  }

  function startProtocol1() {
    // Clear queue and interrupt current audio
    voiceQueue.length = 0;
    if (activeVoiceJob?.audio) {
      activeVoiceJob.audio.pause();
      activeVoiceJob.audio.currentTime = 0;
      activeVoiceJob = null;
    }
    // Set briefing phase (will append a briefing message)
    setPhase('briefing', { skipProtocolVoice: true });
    // Enqueue briefing voice
    const briefing = pickResponse('briefing', 'Campaign Protocol 1 engaged.', audioLibrary.voice.protocol1);
    enqueueVoice(briefing.voice, { priority: voicePriority.protocol });
  }

  function renderPhase() {
    phaseButtons.forEach((btn) => btn.classList.toggle('phase-active', btn.dataset.phase === gameState.phase));
    updatePhaseButtonStates();

    if (gameState.phase === 'briefing') {
      phaseTitle.textContent = 'Mission Briefing';
      phaseContent.innerHTML = `
        <div class="mindweave-card space-y-2">
          <p>Welcome Weaver. You must restore a fractured society using cognitive planning, emotional diplomacy, and reflective learning.</p>
          <ul class="list-disc pl-5 text-slate-300 text-xs space-y-1">
            <li>IQ Engine: Dual N-Back, resource balancing, logic gates.</li>
            <li>EQ Engine: micro-expression reading, empathy dialogue, emotional regulation.</li>
            <li>Debrief: explain strategy, transfer to real-world behavior.</li>
          </ul>
          <div class="mt-2">
            <label class="block mb-1">Choose initial stabilization order:</label>
            <select id="briefing-plan" class="mindweave-input w-full">
              <option value="">Select sequence...</option>
              <option value="wrong">Debrief → EQ → IQ</option>
              <option value="correct">IQ → EQ → Debrief</option>
              <option value="wrong2">EQ → IQ → Debrief</option>
            </select>
          </div>
          <button id="start-campaign" class="mindweave-action">Acknowledge Briefing</button>
        </div>`;
      document.getElementById('start-campaign').onclick = async () => {
        const plan = document.getElementById('briefing-plan').value;
        if (plan !== 'correct') {
          appendChat('SYSTEM', 'Lock briefing by choosing the proper stabilization order first.', 'text-yellow-400');
          playSfx('error');
          return;
        }
        markObjective('briefing_plan', 'Stabilization sequence confirmed');
        markObjective('briefing_read', 'Briefing acknowledged');
        { const ack = pickResponse('briefing', 'Acknowledged. Mission brief locked.', audioLibrary.voice.briefingAcknowledge); appendChat('Architect-7', ack.text, 'text-white', ack.voice); }
        // Removed duplicate Academy suggestion here
        setPhase('iq', { skipProtocolVoice: true });
        enqueueVoice(audioLibrary.voice.protocol1, { priority: voicePriority.sequence });
      };
    }

    if (gameState.phase === 'iq') {
      if (!gameState.nbackSequence.length) gameState.nbackSequence = Array.from({ length: 6 }, () => Math.ceil(Math.random() * 4));
      if (!gameState.threatCodePuzzle) gameState.threatCodePuzzle = generateThreatCodePuzzle();
      phaseTitle.textContent = 'IQ Systems Repair';
      phaseContent.innerHTML = `
        <div class="space-y-3 text-xs">
          <div class="mindweave-card">
            <strong>Dual N-Back (2-back)</strong>
            <p>Sequence: ${gameState.nbackSequence.join(' - ')}</p>
            <p>Enter the value at the final position's 2-back match (the number shown exactly two places before the last number).</p>
            <input id="nback-input" class="mindweave-input w-20" type="number" min="1" max="4" />
            <button id="nback-submit" class="mindweave-action ml-2">Validate</button>
          </div>
          <div class="mindweave-card">
            <strong>Resource Management</strong>
            <p>Allocate power cores A + B to reach exact stabilization target (${gameState.resourceTarget}).</p>
            <input id="resource-a" class="mindweave-input w-16" type="number" placeholder="A" /> +
            <input id="resource-b" class="mindweave-input w-16" type="number" placeholder="B" />
            <button id="resource-submit" class="mindweave-action ml-2">Route</button>
          </div>
          <div class="mindweave-card">
            <strong>Logic Gate Repair</strong>
            <p>Inputs: A=true, B=false. Choose output for (A AND B) OR (NOT B).</p>
            <select id="logic-choice" class="mindweave-input"><option>false</option><option>true</option></select>
            <button id="logic-submit" class="mindweave-action ml-2">Commit</button>
          </div>
          <div class="mindweave-card">
            <strong>Task-Switching Relay</strong>
            <p>Rule A (blue lane): if number is odd add 3, if even double it. For input <span class="text-cyan-300 font-bold">7</span>, transmit output.</p>
            <input id="switch-input" class="mindweave-input w-24" type="number" />
            <button id="switch-submit" class="mindweave-action ml-2">Transmit</button>
          </div>
          <div class="mindweave-card">
            <strong>Planning Optimization</strong>
            <p>You can stabilize only two districts this cycle: Habitat (+6), Transit (+4), MedGrid (+7), WaterNet (+5). Target gain is <span class="text-cyan-300 font-bold">11</span>. Choose the correct pair.</p>
            <select id="planning-choice" class="mindweave-input">
              <option value="">Select pair...</option>
              <option value="habitat-transit">Habitat + Transit</option>
              <option value="transit-water">Transit + WaterNet</option>
              <option value="transit-med">Transit + MedGrid</option>
              <option value="med-water">MedGrid + WaterNet</option>
            </select>
            <button id="planning-submit" class="mindweave-action ml-2">Commit Plan</button>
          </div>
          <div class="mindweave-card">
            <strong>Anomaly Threat Code</strong>
            <p>${gameState.threatCodePuzzle.prompt}</p>
            <input id="threatcode-input" class="mindweave-input w-24" type="number" />
            <button id="threatcode-submit" class="mindweave-action ml-2">Seal Breach</button>
          </div>
        </div>`;

      document.getElementById('nback-submit').onclick = () => {
        const guess = Number(document.getElementById('nback-input').value);
        const expected = gameState.nbackSequence[gameState.nbackSequence.length - 3];
        registerChallengeAttempt(
          'nback_clear',
          (attemptCount) => {
            if (guess !== expected) return false;
            gameState.iqScore = Math.min(100, gameState.iqScore + 12);
            awardProtocolScore('iq', 15, attemptCount);
            markObjective('nback_clear', 'Dual N-Back solved');
            appendChat('SYSTEM', `Working memory lock acquired on attempt ${attemptCount}.`, 'text-emerald-400');
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.iqScore = Math.max(0, gameState.iqScore - 6);
            appendChat('SYSTEM', 'N-back mismatch. Retry with pattern focus.', 'text-yellow-400');
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('resource-submit').onclick = () => {
        const a = Number(document.getElementById('resource-a').value);
        const b = Number(document.getElementById('resource-b').value);
        registerChallengeAttempt(
          'resource_clear',
          (attemptCount) => {
            if (a + b !== gameState.resourceTarget) return false;
            gameState.iqScore = Math.min(100, gameState.iqScore + 10);
            awardProtocolScore('iq', 15, attemptCount);
            markObjective('resource_clear', 'Resource routing stable');
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.iqScore = Math.max(0, gameState.iqScore - 4);
            appendChat('SYSTEM', 'Supply chain imbalance detected.', 'text-yellow-400');
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('logic-submit').onclick = () => {
        const output = document.getElementById('logic-choice').value;
        registerChallengeAttempt(
          'logic_clear',
          (attemptCount) => {
            if (output !== 'true') return false;
            gameState.iqScore = Math.min(100, gameState.iqScore + 10);
            awardProtocolScore('iq', 15, attemptCount);
            markObjective('logic_clear', 'Logic gate repaired');
            playSfx('correct');
            if (isPhaseComplete('iq')) {
              setPhase('eq');
            }
            updateBars();
            return true;
          },
          () => {
            gameState.iqScore = Math.max(0, gameState.iqScore - 4);
            appendChat('SYSTEM', 'Gate output incorrect.', 'text-yellow-400');
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('switch-submit').onclick = () => {
        const output = Number(document.getElementById('switch-input').value);
        registerChallengeAttempt(
          'iq_switch_clear',
          (attemptCount) => {
            if (output !== 10) return false;
            gameState.iqScore = Math.min(100, gameState.iqScore + 8);
            awardProtocolScore('iq', 8, attemptCount);
            markObjective('iq_switch_clear', 'Task-switching relay stabilized');
            { const r = pickResponse('iq_success', 'Cognitive flexibility relay cleared.', null); appendChat('SYSTEM', r.text, 'text-emerald-400'); }
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.iqScore = Math.max(0, gameState.iqScore - 5);
            appendChat('SYSTEM', 'Switching rule drift detected. Re-evaluate Rule A.', 'text-yellow-400');
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('planning-submit').onclick = () => {
        const choice = document.getElementById('planning-choice').value;
        registerChallengeAttempt(
          'iq_planning_clear',
          (attemptCount) => {
            if (choice !== 'transit-med') return false;
            gameState.iqScore = Math.min(100, gameState.iqScore + 8);
            awardProtocolScore('iq', 8, attemptCount);
            markObjective('iq_planning_clear', 'Planning optimization complete');
            { const r = pickResponse('iq_success', 'Executive planning alignment confirmed.', null); appendChat('SYSTEM', r.text, 'text-emerald-400'); }
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.iqScore = Math.max(0, gameState.iqScore - 5);
            appendChat('SYSTEM', 'Planning objective not met. Target gain remains 11.', 'text-yellow-400');
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('threatcode-submit').onclick = () => {
        const code = Number(document.getElementById('threatcode-input').value);
        if (code !== gameState.threatCodePuzzle.answer) {
          gameState.iqScore = Math.max(0, gameState.iqScore - 5);
          appendChat('SYSTEM', 'Threat code mismatch. Recalculate anomaly key.', 'text-yellow-400');
          playSfx('wrong');
          updateBars();
          return;
        }
        gameState.iqScore = Math.min(100, gameState.iqScore + 8);
        awardProtocolScore('iq', 10, 1);
        markObjective('iq_threatcode_clear', 'Anomaly threat code neutralized');
        gameState.threatCodePuzzle = generateThreatCodePuzzle();
        playSfx('correct');
        updateBars();
        if (isPhaseComplete('iq')) setPhase('eq');
      };
    }

    if (gameState.phase === 'eq') {
      phaseTitle.textContent = 'EQ Diplomacy & Regulation';
      phaseContent.innerHTML = `
        <div class="space-y-3 text-xs">
          <div class="mindweave-card">
            <strong>Micro-expression Probe (FACS-inspired)</strong>
            <p>NPC profile: brow raise + lip press + avert gaze. Best emotional interpretation?</p>
            <select id="micro-choice" class="mindweave-input"><option value="fear">Fear / uncertainty</option><option value="joy">Joy</option><option value="disgust">Disgust</option></select>
            <button id="micro-submit" class="mindweave-action ml-2">Analyze</button>
          </div>
          <div class="mindweave-card">
            <strong>Cognitive Reappraisal</strong>
            <p>Architect-7 says: "If we fail this district, every citizen will turn against us." Select the best regulation response.</p>
            <select id="reframe-choice" class="mindweave-input">
              <option value="">Choose response...</option>
              <option value="catastrophe">"You're right, everything is doomed."</option>
              <option value="reframe">"Let's narrow scope: one district is at risk, but we can still recover by sequencing priorities."</option>
              <option value="dismiss">"Stop overreacting and execute."</option>
            </select>
            <button id="reframe-submit" class="mindweave-action mt-2">Apply Reframe</button>
          </div>
          <div class="mindweave-card">
            <strong>Co-Regulation Boundary</strong>
            <p>Select the statement that balances empathy + direction.</p>
            <select id="boundary-choice" class="mindweave-input">
              <option value="">Choose statement...</option>
              <option value="hard">"Feelings are irrelevant. Follow orders."</option>
              <option value="balanced">"I hear the load you're carrying; let's take a 20-second reset, then commit to one actionable next step."</option>
              <option value="avoid">"We'll decide later, let's postpone."</option>
            </select>
            <button id="boundary-submit" class="mindweave-action mt-2">Lock Statement</button>
          </div>
          <div class="mindweave-card">
            <strong>Empathy Bridge</strong>
            <p>Use chat to validate, reflect, and guide Architect-7. Suggestion: "I hear your concern, let's stabilize one subsystem at a time."</p>
            <button id="eq-bridge-submit" class="mindweave-action mt-2">Confirm Active Listening</button>
          </div>
          <div class="mindweave-card">
            <strong>Biometric Regulation</strong>
            <p>Architect sets a BPM. Listen to 4 beats, then reproduce with 4 Regulation Pulse clicks (±5% BPM tolerance).</p>
            <button id="start-regulation-sequence" class="mindweave-action mt-2">Start Pulse Regulation Sequence</button>
          </div>
        </div>`;

      document.getElementById('micro-submit').onclick = () => {
        const value = document.getElementById('micro-choice').value;
        registerChallengeAttempt(
          'micro_clear',
          (attemptCount) => {
            if (value !== 'fear') return false;
            gameState.eqScore = Math.min(100, gameState.eqScore + 8);
            awardProtocolScore('eq', 20, attemptCount);
            markObjective('micro_clear', 'Micro-expression recognized');
            updateNPCState('calm', 'Validated and understood');
            { const calm = pickResponse('calm', 'Your empathy parameters are acceptable. My logic loops are stabilizing. Proceed with the temporal hack.', audioLibrary.voice.calmResponse); appendChat('Architect-7', calm.text, 'text-white', calm.voice); }
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.eqScore = Math.max(0, gameState.eqScore - 8);
            updateNPCState('stress', 'Misread social cue');
            { const stress = pickResponse('stress', 'Your aggressive syntax triggers my defense subroutines! The grid cannot be forced!', audioLibrary.voice.stressResponse); appendChat('Architect-7', stress.text, 'text-white', stress.voice); }
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('reframe-submit').onclick = () => {
        const value = document.getElementById('reframe-choice').value;
        registerChallengeAttempt(
          'eq_reframe_clear',
          (attemptCount) => {
            if (value !== 'reframe') return false;
            gameState.eqScore = Math.min(100, gameState.eqScore + 7);
            awardProtocolScore('eq', 7, attemptCount);
            markObjective('eq_reframe_clear', 'Cognitive reappraisal executed');
            { const r = pickResponse('eq_success', 'Threat load reduced. Reframing restored my planning bandwidth.', null); appendChat('Architect-7', r.text, 'text-white'); }
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.eqScore = Math.max(0, gameState.eqScore - 6);
            { const r = pickResponse('eq_repair', 'Reappraisal failed. Catastrophic framing remains active.', null); appendChat('SYSTEM', r.text, 'text-yellow-400'); }
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('boundary-submit').onclick = () => {
        const value = document.getElementById('boundary-choice').value;
        registerChallengeAttempt(
          'eq_boundary_clear',
          (attemptCount) => {
            if (value !== 'balanced') return false;
            gameState.eqScore = Math.min(100, gameState.eqScore + 7);
            awardProtocolScore('eq', 8, attemptCount);
            markObjective('eq_boundary_clear', 'Co-regulation boundary aligned');
            { const r = pickResponse('eq_success', 'Boundary accepted. I can regulate and execute simultaneously.', null); appendChat('Architect-7', r.text, 'text-white'); }
            playSfx('correct');
            updateBars();
            return true;
          },
          () => {
            gameState.eqScore = Math.max(0, gameState.eqScore - 6);
            { const r = pickResponse('eq_repair', 'Boundary response was either avoidant or coercive. Retry.', null); appendChat('SYSTEM', r.text, 'text-yellow-400'); }
            playSfx('wrong');
            updateBars();
          }
        );
      };

      document.getElementById('eq-bridge-submit').onclick = () => {
        if (gameState.scoring.chatSupports < 2) {
          appendChat('SYSTEM', 'Use chat to send at least two active-listening responses first.', 'text-yellow-400');
          playSfx('error');
          return;
        }
        markObjective('eq_bridge_clear', 'Active listening confirmation logged');
        appendChat('SYSTEM', 'Empathy bridge stabilized.', 'text-emerald-400');
        playSfx('correct');
      };

      document.getElementById('start-regulation-sequence').onclick = async () => {
        if (gameState.pulseSequence.active) return;
        const targetBpm = Math.floor(Math.random() * 36) + 72;
        gameState.pulseSequence = {
          ...gameState.pulseSequence,
          targetBpm,
          tapCount: 0,
          lastTapTime: 0,
          tapIntervals: [],
          awaitingInput: false,
          active: true,
        };
        appendChat('Architect-7', `Pulse protocol engaged. Match my rhythm at ${targetBpm} BPM.`, 'text-white');
        await playArchitectPulseSequence(targetBpm);
        gameState.pulseSequence.awaitingInput = true;
        appendChat('SYSTEM', 'Your turn: click Regulation Pulse four times to reproduce the BPM.', 'text-sky-300');
      };
    }

    if (gameState.phase === 'debrief') {
      phaseTitle.textContent = 'Metacognitive Debrief';
      phaseContent.innerHTML = `
        <div class="mindweave-card text-xs space-y-3">
          <p>Write 2-4 sentences explaining (1) your in-game strategy, (2) one moment where you adapted, and (3) how you'll apply that approach in real teamwork.</p>
          <textarea id="debrief-text" class="mindweave-input w-full h-28" placeholder="Example: I slowed my pace, checked the target conditions, and corrected mistakes quickly. When Architect-7 escalated, I switched to validation before directives. In team meetings, I'll pause, confirm shared goals, then propose one concrete next step."></textarea>
          <button id="debrief-submit" class="mindweave-action">Submit Debrief</button>
          <label class="block">List two concrete transfer commitments (one per line, each starts with an action verb):</label>
          <textarea id="debrief-commitment" class="mindweave-input w-full h-20" placeholder="1) Pause before reacting in conflict\n2) Validate teammate concerns before proposing fixes"></textarea>
        </div>`;
      document.getElementById('debrief-submit').onclick = async () => {
        const reflection = document.getElementById('debrief-text').value.trim();
        const commitments = document.getElementById('debrief-commitment').value.trim().split('\n').map((line) => line.trim()).filter(Boolean);
        if (reflection.length < 20) {
          appendChat('SYSTEM', 'Debrief too short. Submit at least 20 characters covering strategy, adaptation, and real-world transfer.', 'text-yellow-400');
          { const debriefRetry = pickResponse('debrief', 'Debrief received, but include explicit strategy and transfer language for stronger consolidation.', audioLibrary.voice.debriefReceived); appendChat('Architect-7', debriefRetry.text, 'text-white', debriefRetry.voice); }
          playSfx('error');
          return;
        }
        if (commitments.length < 2) {
          appendChat('SYSTEM', 'Include at least two transfer commitments.', 'text-yellow-400');
          playSfx('error');
          return;
        }
        markObjective('debrief_submit', 'Debrief submitted');
        markObjective('debrief_commitment', 'Transfer commitments logged');
        const strategyTerms = /(strategy|plan|sequence|prioritize|check|adapt|monitor)/i.test(reflection);
        const transferTerms = /(team|work|meeting|conflict|project|real-world|colleague)/i.test(reflection);
        const adaptationTerms = /(adapt|switch|adjust|reframe|changed|correct)/i.test(reflection);
        const qualityBonus = [strategyTerms, transferTerms, adaptationTerms].filter(Boolean).length;
        const lengthBand = reflection.length >= 220 ? 6 : reflection.length >= 140 ? 4 : 2;
        gameState.protocolScore.debrief = Math.min(protocolWeights.debrief, 5 + qualityBonus + lengthBand);
        renderFinalScore();
        await sendTaskMessage(`Debrief reflection: ${reflection}`, 'debrief_reflection');
        appendChat('SYSTEM', 'Campaign loop complete. Far-transfer reinforcement logged.', 'text-emerald-400');
        maybeShowEndgameModal();
      };
    }

    renderTelemetry();
  }

  function resetGameState() {
    gameState.phase = 'briefing';
    gameState.iqScore = 48;
    gameState.eqScore = 82;
    gameState.progress = 0;
    gameState.completedObjectives = new Set();
    gameState.nbackSequence = [];
    gameState.regulationBreaths = 0;
    gameState.resourceTarget = Math.floor(Math.random() * 13) + 8;
    gameState.threatCodePuzzle = null;
    gameState.challengeAttempts = {
      nback_clear: { count: 0, locked: false },
      resource_clear: { count: 0, locked: false },
      logic_clear: { count: 0, locked: false },
      iq_switch_clear: { count: 0, locked: false },
      iq_planning_clear: { count: 0, locked: false },
      micro_clear: { count: 0, locked: false },
      eq_reframe_clear: { count: 0, locked: false },
      eq_boundary_clear: { count: 0, locked: false },
    };
    gameState.pulseSequence = {
      targetBpm: null,
      tolerance: 0.05,
      tapCount: 0,
      lastTapTime: 0,
      tapIntervals: [],
      awaitingInput: false,
      active: false,
    };
    gameState.protocolScore = { iq: 0, eq: 0, debrief: 0 };
    gameState.scoring = { errors: 0, lockedProtocols: 0, chatSupports: 0, completedRun: false };

    closeEndgameModal();
    chatHistory.innerHTML = '';
    updateNPCState('neutral', 'Session reset and synchronized');
    updateBars();
    renderObjectives();
    startProtocol1();
  }

  function setPhase(phase, options = {}) {
    const { skipProtocolVoice = false } = options;
    if (!isPhaseAccessible(phase)) {
      appendChat('SYSTEM', `Access denied: complete ${orderedPhases[orderedPhases.indexOf(phase) - 1].toUpperCase()} protocol first.`, 'text-yellow-400');
      playSfx('error');
      return;
    }
    gameState.phase = phase;
    renderPhase();
    const phaseGroup = phase === 'iq' ? 'IQ' : phase === 'eq' ? 'EQ' : phase;
    if (['briefing', 'IQ', 'EQ', 'debrief'].includes(phaseGroup)) {
      const protocol = pickResponse(phaseGroup, `Protocol update: ${phaseGroup} engaged.`, null);
      appendChat('Architect-7', protocol.text, 'text-white');
      if (!skipProtocolVoice && protocol.voice) {
        enqueueVoice(protocol.voice, { priority: voicePriority.protocol });
      }
    }
  }

  async function handleChatSubmit() {
    const text = chatInput.value.trim();
    if (!text) return;
    appendChat('Weaver', text);
    chatInput.value = '';
    updateNPCState('thinking', 'Processing Semantics...');
    await sendTaskMessage(text, gameState.phase === 'debrief' ? 'debrief_reflection' : 'npc_dialogue');

    if (gameState.phase === 'eq' && /understand|hear|support|calm|together|validate|breathe|reset/i.test(text)) {
      gameState.scoring.chatSupports += 1;
      gameState.eqScore = Math.min(100, gameState.eqScore + 3);
      updateBars();
      if (gameState.completedObjectives.has('micro_clear') && gameState.completedObjectives.has('regulation_clear') && gameState.completedObjectives.has('eq_bridge_clear')) {
        setPhase('debrief');
      }
    }
  }

  sendBtn.addEventListener('click', handleChatSubmit);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatSubmit();
  });

  document.getElementById('btn-empathy-ping').addEventListener('click', () => {
    if (audioState.sfxMuted === false) {
      const ping = createAudio(audioLibrary.sfx.ping);
      ping.volume = audioState.sfxVolume;
      ping.play().catch(() => {});
    }
    appendChat('SYSTEM', '[Empathy Ping] Architect-7 is masking fear and resource shame. Use validation before directives.', 'text-slate-400 italic');
  });

  document.getElementById('btn-reset-iq').addEventListener('click', () => {
    gameState.nbackSequence = [];
    gameState.resourceTarget = Math.floor(Math.random() * 13) + 8;
    gameState.threatCodePuzzle = generateThreatCodePuzzle();
    gameState.challengeAttempts.nback_clear = { count: 0, locked: false };
    gameState.challengeAttempts.resource_clear = { count: 0, locked: false };
    gameState.challengeAttempts.logic_clear = { count: 0, locked: false };
    gameState.challengeAttempts.iq_switch_clear = { count: 0, locked: false };
    gameState.challengeAttempts.iq_planning_clear = { count: 0, locked: false };
    gameState.iqScore = 48;
    playSfx('powerDown');
    updateBars();
    if (gameState.phase === 'iq') renderPhase();
  });

  document.getElementById('btn-regulate').addEventListener('click', () => {
    if (gameState.phase !== 'eq') {
      appendChat('SYSTEM', 'Regulation Pulse is only available in EQ mode.', 'text-yellow-400');
      playSfx('error');
      return;
    }
    if (!gameState.pulseSequence.awaitingInput) {
      appendChat('SYSTEM', 'Start the pulse regulation sequence first, then mirror the Architect BPM.', 'text-yellow-400');
      playSfx('error');
      return;
    }
    const now = performance.now();
    playPulseBeat();
    gameState.pulseSequence.tapCount += 1;
    if (gameState.pulseSequence.lastTapTime) {
      gameState.pulseSequence.tapIntervals.push(now - gameState.pulseSequence.lastTapTime);
    }
    gameState.pulseSequence.lastTapTime = now;
    gameState.regulationBreaths = Math.min(4, gameState.pulseSequence.tapCount);
    appendChat('SYSTEM', `Regulation pulse ${gameState.pulseSequence.tapCount}/4 logged.`, 'text-sky-300');

    if (gameState.pulseSequence.tapCount >= 4) {
      const averageInterval = gameState.pulseSequence.tapIntervals.reduce((sum, item) => sum + item, 0) / gameState.pulseSequence.tapIntervals.length;
      const playerBpm = Math.round(60000 / averageInterval);
      const targetBpm = gameState.pulseSequence.targetBpm;
      const tolerance = targetBpm * gameState.pulseSequence.tolerance;
      const difference = Math.abs(playerBpm - targetBpm);
      if (difference <= tolerance) {
        gameState.eqScore = Math.min(100, gameState.eqScore + 8);
        gameState.protocolScore.eq = Math.min(protocolWeights.eq, gameState.protocolScore.eq + 15);
        markObjective('regulation_clear', `Pulse rhythm matched (${playerBpm} BPM vs ${targetBpm} BPM)`);
        appendChat('SYSTEM', `Pulse match success. Your BPM ${playerBpm} is within ±5% of ${targetBpm}.`, 'text-emerald-400');
        playSfx('correct');
      } else {
        gameState.eqScore = Math.max(0, gameState.eqScore - 6);
        appendChat('SYSTEM', `Pulse mismatch. Your BPM ${playerBpm} diverged from ${targetBpm}. Retry sequence.`, 'text-yellow-400');
        playSfx('wrong');
      }
      gameState.pulseSequence.awaitingInput = false;
      gameState.pulseSequence.active = false;
      gameState.pulseSequence.tapCount = 0;
      gameState.pulseSequence.lastTapTime = 0;
      gameState.pulseSequence.tapIntervals = [];
      renderFinalScore();
      if (gameState.phase === 'eq' && isPhaseComplete('eq')) setPhase('debrief');
    }
    updateBars();
    renderTelemetry();
  });

  settingsClose.addEventListener('click', closeSettings);
  endgameClose.addEventListener('click', closeEndgameModal);
  endgameRestart.addEventListener('click', () => {
    closeEndgameModal();
    resetGameState();
  });
  endgameModal.addEventListener('click', (event) => {
    if (event.target === endgameModal) closeEndgameModal();
  });
  settingsModal.addEventListener('click', (event) => {
    if (event.target === settingsModal) closeSettings();
  });

  bgmVolumeInput.addEventListener('input', (event) => {
    audioState.bgmVolume = Number(event.target.value);
    applyAudioSettings();
  });

  sfxVolumeInput.addEventListener('input', (event) => {
    audioState.sfxVolume = Number(event.target.value);
  });

  bgmMuteInput.addEventListener('change', (event) => {
    audioState.bgmMuted = event.target.checked;
    applyAudioSettings();
  });

  sfxMuteInput.addEventListener('change', (event) => {
    audioState.sfxMuted = event.target.checked;
  });

  voiceVolumeInput.addEventListener('input', (event) => {
    audioState.voiceVolume = Number(event.target.value);
    applyAudioSettings();
  });

  voiceMuteInput.addEventListener('change', (event) => {
    audioState.voiceMuted = event.target.checked;
    if (audioState.voiceMuted) {
      if (activeVoiceJob?.audio) {
        activeVoiceJob.audio.pause();
        activeVoiceJob.audio.currentTime = 0;
      }
      activeVoiceJob = null;
      voiceQueue.length = 0;
    }
    applyAudioSettings();
  });

  phaseButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const phase = btn.dataset.phase;
      if (!isPhaseAccessible(phase)) {
        appendChat('SYSTEM', 'Complete the current protocol before advancing.', 'text-yellow-400');
        playSfx('error');
        return;
      }
      setPhase(phase, { skipProtocolVoice: phase === 'briefing' });
      if (phase === 'briefing') {
        enqueueVoice(audioLibrary.voice.protocol1, { priority: voicePriority.protocol, clearQueue: true, interrupt: true });
      }
    });
  });

  await loadResponseTemplates();

  cacheSfx();
  bgmVolumeInput.value = String(audioState.bgmVolume);
  sfxVolumeInput.value = String(audioState.sfxVolume);
  voiceVolumeInput.value = String(audioState.voiceVolume);
  bgmMuteInput.checked = audioState.bgmMuted;
  sfxMuteInput.checked = audioState.sfxMuted;
  voiceMuteInput.checked = audioState.voiceMuted;
  startBgm();

  ensureMindweaveSelected().catch((error) => {
    appendChat('SYSTEM', `Backend initialization error: ${error.message}`, 'text-red-500');
    apiStatusEl.textContent = 'Backend link unstable.';
    apiStatusEl.classList.replace('text-green-400', 'text-red-500');
  });

  // Always play the full startup sequence on every page load
  await playStartupSequence();

  updateBars();
  renderObjectives();
  renderFinalScore();

  const launchParams = new URLSearchParams(window.location.search);
  if (launchParams.get('start') === 'home') {
    launchProtocolOneFromHome();
  }

  window.addEventListener('resize', () => {
    const nextAspect = window.innerWidth / window.innerHeight;
    camera.left = -frustumSize * nextAspect / 2;
    camera.right = frustumSize * nextAspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = -frustumSize / 2;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();
    controls.update();

    gridGroup.children.forEach((mesh, index) => {
      mesh.position.y = mesh.userData.baseY + Math.sin(time * 2 + index * 0.1) * 0.1;
    });

    npcMesh.rotation.y += 0.01;
    npcMesh.rotation.x += 0.005;
    npcMesh.position.y = 3 + Math.sin(time * 1.5) * 0.3;
    if (currentEmotion === 'stress') {
      npcMesh.position.x = (Math.random() - 0.5) * 0.1;
      npcMesh.scale.setScalar(1 + Math.sin(time * 20) * 0.05);
    } else {
      npcMesh.position.x = 0;
      npcMesh.scale.setScalar(1);
    }

    renderer.render(scene, camera);
  }

  animate();
});