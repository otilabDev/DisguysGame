import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ---------- Assets ----------
const SCENE_URL = new URL('../scene/main-menu.glb', import.meta.url).href;

// ---------- Camera drift ----------
// The Unity camera is static. A slow figure-eight keeps the shot alive without turning it
// into a fly-through.
const DRIFT_X     = 0;
const DRIFT_Y     = 0;
const DRIFT_SPEED = 0.11;
const PORTRAIT_FOV_BOOST = 14;   // narrow screens crop the sides away

// ---------- Sky ----------
// Doubles as the environment probe, so the stops below the horizon are what bounces back up
// onto the undersides.
const SKY_STOPS = [
  [0.00, '#25283a'],
  [0.40, '#454a63'],
  [0.60, '#6c7190'],
  [0.74, '#888ba8'],
  [1.00, '#3b3d50'],
];

// ---------- Grade ----------
// The threshold sits just under the warm horizon, so the sky glows and the geometry does not.
const BLOOM_STRENGTH  = 0.48;
const BLOOM_RADIUS    = 0.46;
const BLOOM_THRESHOLD = 0.68;
const EXPOSURE = 1.24;
const MSAA_SAMPLES = 2;
const VIGNETTE   = 0.16;
const SATURATION = 1.08;
const CONTRAST = 0.94;
const CHROMATIC_ABERRATION = 0;
const FILM_GRAIN = 0;
const VIGNETTE_COLOR = [0.24, 0.25, 0.34];
const SHADOW_TINT    = [0.50, 0.50, 0.78];
const HIGHLIGHT_TINT = [1.00, 0.84, 0.62];
const TINT_AMOUNT    = 0.16;
const SSAO = { kernelRadius: 12, minDistance: 0.002, maxDistance: 0.12 };

// ---------- Scene object overrides ----------
const OBJECT_EDITS = {};

// ---------- Depth of field ----------
const DOF = { on: false, aperture: 0, maxBlur: 0 };

// ---------- Dust ----------
const DUST_COUNT    = 0;
const DUST_AREA     = [16, 7, 20];   // x, y, z spread around the shot
const DUST_CENTER   = [0, 3.0, -8];   // the camera looks down -Z from the origin
const DUST_SIZE     = 0.055;
const DUST_RISE     = 0.18;    // units per second
const DUST_COLOR    = 0xdcf2ff;

// ---------- Pointer interaction ----------
const INTERACTION = {
  wobbleDegrees: 5,
  spring: 72,
  damping: 12,
  particleCount: 160,
  particlesPerBurst: 7,
  particleIntervalMs: 38,
};
const PROP_RUNNERS = {
  count: 3,
  speedMin: 1.7,
  speedMax: 2.5,
  turnSpeed: 6,
  clearance: 0.7,
  hopHeight: 0.12,
  navigationStep: 4,
};
const CAMERA_CHAPTERS = [
  { id: 'top', offset: [0, 0, 0], rotation: [0, 0, 0] },
  { id: 'about', offset: [0.34, 0.08, 0.24], rotation: [-0.5, 1.4, 0] },
  { id: 'about-overview', offset: [-0.46, 0.12, 0.42], rotation: [0.4, -1.6, 0] },
  { id: 'gameplay', offset: [0.4, 0.06, 0.28], rotation: [-0.2, 1.8, 0] },
  { id: 'screenshots', offset: [-0.3, -0.05, 0.5], rotation: [0.5, -1.4, 0] },
  { id: 'roadmap', offset: [0.22, 0.1, 0.38], rotation: [-0.3, 1.1, 0] },
  { id: 'social', offset: [0, 0, 0.58], rotation: [0, 0, 0] },
];
// ---------- Lighting ----------
// The exported directional lights sit far out, where the scene's own origin is. Direction is
// all a directional light needs, so they get pulled in to where a shadow camera can work.
const LIGHT_DISTANCE = 30;
const SHADOW_EXTENT  = 24;
const ENV = { intensity: 1.45 };
const AMBIENT = { sky: 0x9299e8, ground: 0x3d3d57, intensity: 1.35 };
const LIGHT_PRESET = [
  { color: 0xffe8bd, azimuth: -45, elevation: 52, intensity: 1.85,
    shadowSoftness: 5.5, shadowBias: -0.0005, shadowNormalBias: 0.035 },
  { color: 0x8c96eb, azimuth: 135, elevation: 28, intensity: 0.55 },
];
const FOG = { color: 0x45475d, density: 0.012 };


// BokehPass builds its depth buffer by re-rendering the scene, which drags in two things that
// have no business being there. The sky background is drawn as a unit box at the camera, so it
// would report near depth and come back sharp. The dust would punch sharp specks through the
// blurred distance. Both are hidden for that pass only.
class SceneDepthBokehPass extends BokehPass {
  constructor(scene, camera, params, hidden) {
    super(scene, camera, params);
    this.hidden = hidden;
  }

  render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
    const background = this.scene.background;
    const wasVisible = this.hidden.map((o) => o.visible);
    this.scene.background = null;
    for (const o of this.hidden) o.visible = false;

    super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);

    this.scene.background = background;
    this.hidden.forEach((o, i) => { o.visible = wasVisible[i]; });
  }
}

// Runs after OutputPass, so it grades display-referred colour: cool shadows, warm highlights,
// then saturation and vignette.
const GradeShader = {
  uniforms: {
    tDiffuse:       { value: null },
    uVignette:      { value: VIGNETTE },
    uSaturation:    { value: SATURATION },
    uContrast:      { value: CONTRAST },
    uChromaticAberration: { value: CHROMATIC_ABERRATION },
    uFilmGrain:     { value: FILM_GRAIN },
    uVignetteColor: { value: new THREE.Vector3(...VIGNETTE_COLOR) },
    uTime:          { value: 0 },
    uShadowTint:    { value: new THREE.Vector3(...SHADOW_TINT) },
    uHighlightTint: { value: new THREE.Vector3(...HIGHLIGHT_TINT) },
    uTintAmount:    { value: TINT_AMOUNT },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uVignette;
    uniform float uSaturation;
    uniform float uContrast;
    uniform float uChromaticAberration;
    uniform float uFilmGrain;
    uniform vec3 uVignetteColor;
    uniform float uTime;
    uniform vec3  uShadowTint;
    uniform vec3  uHighlightTint;
    uniform float uTintAmount;
    varying vec2 vUv;

    void main() {
      vec2 d = vUv - 0.5;
      float radial = dot(d, d);
      vec2 direction = normalize(d + vec2(0.00001));
      vec2 offset = direction * radial * uChromaticAberration * 0.006;
      vec4 texel = texture2D(tDiffuse, vUv);
      vec3 color = vec3(
        texture2D(tDiffuse, vUv + offset).r,
        texel.g,
        texture2D(tDiffuse, vUv - offset).b
      );

      float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
      vec3 tint = mix(uShadowTint, uHighlightTint, smoothstep(0.15, 0.85, luma));
      color = mix(color, color * tint * 2.0, uTintAmount);

      color = mix(vec3(luma), color, uSaturation);
      color = (color - 0.5) * uContrast + 0.5;

      float vignette = smoothstep(0.18, 0.72, length(d));
      color = mix(color, color * uVignetteColor, vignette * uVignette);

      float grain = fract(sin(dot(vUv * vec2(1920.0, 1080.0) + uTime,
        vec2(12.9898, 78.233))) * 43758.5453);
      color += (grain - 0.5) * uFilmGrain * 0.08;

      gl_FragColor = vec4(color, texel.a);
    }
  `,
};

const container = document.getElementById('intro');
const loadingEl = document.getElementById('loading');
const prefersReducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
export async function initScene() {
  // ---------- Loaders ----------
  const manager = new THREE.LoadingManager();
  let loadFailed = false;
  manager.onProgress = (url, loaded, total) => {
    if (loadFailed) return;
    loadingEl.textContent = `Yükleniyor ${Math.round((loaded / total) * 100)}%`;
  };

  const gltfLoader = new GLTFLoader(manager);
  gltfLoader.setMeshoptDecoder(MeshoptDecoder);
  const loadGLTF = (url) => new Promise((res, rej) => gltfLoader.load(url, res, undefined, rej));

  const sceneGltf = await loadGLTF(SCENE_URL).catch((error) => {
    loadFailed = true;
    throw error;
  });

  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = EXPOSURE;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const sky = skyTexture();
  scene.background = sky;
  scene.fog = new THREE.FogExp2(FOG.color, FOG.density);
  scene.add(new THREE.HemisphereLight(AMBIENT.sky, AMBIENT.ground, AMBIENT.intensity));

  // The sky is also the environment probe. That replaces the old flat ambient: roughness and
  // metalness now read on the surfaces instead of every face getting the same wash.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromEquirectangular(sky).texture;
  pmrem.dispose();

  const dust = makeDust();
  scene.add(dust.points);
  const particleScene = new THREE.Scene();
  const contactParticles = makeContactParticles();
  particleScene.add(contactParticles.mesh);

  let camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.3, 1000);
  const camBase = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), fov: 60 };
  const storyPosition = new THREE.Vector3();
  const storyQuaternion = new THREE.Quaternion();
  const storyTargetPosition = new THREE.Vector3();
  const storyTargetQuaternion = new THREE.Quaternion();
  const storyLocalOffset = new THREE.Vector3();
  const storyToOffset = new THREE.Vector3();
  const storyLocalRotation = new THREE.Quaternion();
  const storyEuler = new THREE.Euler();
  const cameraChapters = CAMERA_CHAPTERS.map((chapter) => ({
    ...chapter,
    element: document.getElementById(chapter.id),
  }));
  let cameraStoryReady = false;

  function updateStoryCamera(dt) {
    if (!cameraStoryReady) return;
    let from = cameraChapters[0];
    let to = from;
    let progress = 0;

    if (!prefersReducedMotion) {
      const anchor = scrollY + innerHeight * .46;
      for (let index = 0; index < cameraChapters.length - 1; index += 1) {
        const current = cameraChapters[index];
        const next = cameraChapters[index + 1];
        if (anchor < next.element.offsetTop) {
          from = current;
          to = next;
          const span = Math.max(1, next.element.offsetTop - current.element.offsetTop);
          const linear = Math.min(1, Math.max(0, (anchor - current.element.offsetTop) / span));
          progress = linear * linear * (3 - 2 * linear);
          break;
        }
        from = next;
        to = next;
      }
    }

    storyLocalOffset.fromArray(from.offset).lerp(storyToOffset.fromArray(to.offset), progress);
    storyTargetPosition.copy(storyLocalOffset).applyQuaternion(camBase.quat).add(camBase.pos);

    const rotation = from.rotation.map((value, index) => THREE.MathUtils.lerp(value, to.rotation[index], progress));
    storyEuler.set(...rotation.map((value) => THREE.MathUtils.degToRad(value)));
    storyLocalRotation.setFromEuler(storyEuler);
    storyTargetQuaternion.copy(camBase.quat).multiply(storyLocalRotation);

    const ease = 1 - Math.exp(-dt * 4.5);
    storyPosition.lerp(storyTargetPosition, ease);
    storyQuaternion.slerp(storyTargetQuaternion, ease);
    camera.position.copy(storyPosition);
    camera.quaternion.copy(storyQuaternion);
  }

  // A composer renders into its own targets, so two things the renderer would normally handle
  // stop applying. Its antialias flag is ignored, because that only covers the default
  // framebuffer, so the target has to be multisampled itself. And its pixel ratio is read once
  // at construction, which happens before layout() sets it, so it has to be handed over there
  // as well. Left alone, the scene comes out aliased and upscaled from a half-size buffer.
  const composerTarget = new THREE.WebGLRenderTarget(innerWidth, innerHeight, {
    type: THREE.HalfFloatType,
    samples: MSAA_SAMPLES,
  });
  const composer = new EffectComposer(renderer, composerTarget);
  composer.addPass(new RenderPass(scene, camera));

  const ssao = new SSAOPass(scene, camera, innerWidth, innerHeight);
  ssao.kernelRadius = SSAO.kernelRadius;
  ssao.minDistance = SSAO.minDistance;
  ssao.maxDistance = SSAO.maxDistance;
  composer.addPass(ssao);

  const bokeh = new SceneDepthBokehPass(scene, camera,
    { focus: 5, aperture: DOF.aperture, maxblur: DOF.maxBlur }, [dust.points]);
  bokeh.enabled = DOF.on;
  composer.addPass(bokeh);
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight),
    BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const grade = new ShaderPass(GradeShader);
  composer.addPass(grade);

  function layout() {
    const aspect = innerWidth / innerHeight;
    camera.aspect = aspect;
    camera.fov = aspect < 0.95 ? camBase.fov + PORTRAIT_FOV_BOOST : camBase.fov;
    camera.updateProjectionMatrix();
    const ratio = Math.min(devicePixelRatio, 1.5);
    renderer.setPixelRatio(ratio);
    renderer.setSize(innerWidth, innerHeight);
    composer.setPixelRatio(ratio);
    composer.setSize(innerWidth, innerHeight);
  }

  // Soft round sprite for the dust motes; a texture beats a square point.
  function moteTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.35, 'rgba(255,255,255,0.55)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 64, 64);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  // Motes drift up and sway; each one carries its own phase so the field never pulses in step.
  function makeDust() {
    const pos = new Float32Array(DUST_COUNT * 3);
    const phase = new Float32Array(DUST_COUNT);
    const speed = new Float32Array(DUST_COUNT);
    for (let i = 0; i < DUST_COUNT; i++) {
      pos[i * 3]     = DUST_CENTER[0] + (Math.random() - 0.5) * DUST_AREA[0];
      pos[i * 3 + 1] = DUST_CENTER[1] + (Math.random() - 0.5) * DUST_AREA[1];
      pos[i * 3 + 2] = DUST_CENTER[2] + (Math.random() - 0.5) * DUST_AREA[2];
      phase[i] = Math.random() * Math.PI * 2;
      speed[i] = 0.5 + Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: DUST_SIZE,
      map: moteTexture(),
      color: DUST_COLOR,
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    }));
    points.frustumCulled = false;
    points.renderOrder = 3;
    return { points, pos, phase, speed, geo };
  }

  function skyTexture() {
    const c = document.createElement('canvas');
    c.width = 8; c.height = 512;
    const ctx = c.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, 512);
    for (const [at, color] of SKY_STOPS) g.addColorStop(at, color);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 8, 512);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
  }


  function applyObjectEdits(root) {
    for (const [name, edit] of Object.entries(OBJECT_EDITS)) {
      const target = root.getObjectByName(name);
      if (!target) {
        console.error(`[scene] OBJECT_EDITS names "${name}", which is not in the set`);
        continue;
      }
      if (edit.position) target.position.fromArray(edit.position);
      if (edit.rotation) {
        target.rotation.set(...edit.rotation.map((d) => THREE.MathUtils.degToRad(d)));
      }
      if (edit.scale) target.scale.fromArray(edit.scale);
      if (edit.visible !== undefined) target.visible = edit.visible;
    }
  }

  // The export carries no shadow flags of its own, and the set is what the actor is lit
  // against, so every mesh in it both casts and receives.
  function castAndReceiveShadows(root) {
    root.traverse((c) => {
      if (!c.isMesh) return;
      c.castShadow = true;
      c.receiveShadow = true;
    });
  }

  const MAPS = ['map', 'roughnessMap', 'metalnessMap', 'normalMap', 'emissiveMap', 'aoMap'];

  function prepareMaterials(root) {
    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    root.traverse((c) => {
      if (!c.isMesh) return;
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      for (const m of mats) {
        if (!m.isMeshStandardMaterial) continue;
        m.envMapIntensity = ENV.intensity;
        for (const slot of MAPS) {
          const tex = m[slot];
          if (tex && tex.anisotropy < maxAnisotropy) {
            tex.anisotropy = maxAnisotropy;
            tex.needsUpdate = true;
          }
        }
      }
    });
  }

  // Pulls a directional light in from the sub-scene's far origin, keeping its direction, and
  // gives the first one a shadow camera sized to the visible set.
  function retargetLight(light, withShadow) {
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(light.getWorldQuaternion(new THREE.Quaternion()));
    scene.attach(light);
    light.position.copy(dir).multiplyScalar(-LIGHT_DISTANCE);
    light.target.position.set(0, 0, 0);
    scene.add(light.target);
    if (!withShadow) return;
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    light.shadow.bias = -0.0008;
    light.shadow.normalBias = 0.02;
    const c = light.shadow.camera;
    c.left = -SHADOW_EXTENT; c.right = SHADOW_EXTENT;
    c.top = SHADOW_EXTENT;   c.bottom = -SHADOW_EXTENT;
    c.near = 0.5; c.far = LIGHT_DISTANCE * 2.5;
    c.updateProjectionMatrix();
  }

  function applyLightingPreset(lights) {
    if (lights.length < LIGHT_PRESET.length) {
      throw new Error(`cartoon lighting requires ${LIGHT_PRESET.length} directional lights`);
    }
    LIGHT_PRESET.forEach((preset, index) => {
      const light = lights[index];
      light.color.setHex(preset.color);
      light.intensity = preset.intensity;
      applyDirection(light, preset);
      if (index === 0) {
        light.shadow.radius = preset.shadowSoftness;
        light.shadow.bias = preset.shadowBias;
        light.shadow.normalBias = preset.shadowNormalBias;
      }
    });
  }

  function applyDirection(light, state) {
    const azimuth = THREE.MathUtils.degToRad(state.azimuth);
    const elevation = THREE.MathUtils.degToRad(state.elevation);
    const radius = LIGHT_DISTANCE * Math.cos(elevation);
    light.position.set(
      Math.sin(azimuth) * radius,
      Math.sin(elevation) * LIGHT_DISTANCE,
      Math.cos(azimuth) * radius,
    ).add(light.target.position);
  }

  function makeContactParticles() {
    const geometry = new THREE.TetrahedronGeometry(1, 0);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
      depthTest: false,
      depthWrite: false,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, INTERACTION.particleCount);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.renderOrder = 4;

    const particles = Array.from({ length: INTERACTION.particleCount }, () => ({
      position: new THREE.Vector3(),
      velocity: new THREE.Vector3(),
      rotation: new THREE.Vector3(),
      spin: new THREE.Vector3(),
      life: 0,
      maxLife: 1,
      size: 0,
    }));
    const dummy = new THREE.Object3D();
    const random = new THREE.Vector3();
    let cursor = 0;

    for (let i = 0; i < particles.length; i++) {
      dummy.scale.setScalar(0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }

    function burst(point, normal) {
      for (let i = 0; i < INTERACTION.particlesPerBurst; i++) {
        const index = cursor++ % particles.length;
        const particle = particles[index];
        particle.position.copy(point).addScaledVector(normal, 0.035);
        random.set(Math.random() - 0.5, Math.random() - 0.15, Math.random() - 0.5);
        particle.velocity.copy(normal).multiplyScalar(0.24 + Math.random() * 0.2)
          .addScaledVector(random, 0.28);
        particle.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
        particle.spin.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
          .multiplyScalar(8);
        particle.maxLife = 0.75 + Math.random() * 0.35;
        particle.life = particle.maxLife;
        particle.size = 0.034 + Math.random() * 0.034;
      }
    }

    function update(dt) {
      for (let i = 0; i < particles.length; i++) {
        const particle = particles[i];
        if (particle.life > 0) {
          particle.life = Math.max(0, particle.life - dt);
          particle.velocity.y -= 0.7 * dt;
          particle.position.addScaledVector(particle.velocity, dt);
          particle.rotation.addScaledVector(particle.spin, dt);
        }
        const scale = particle.size * Math.pow(particle.life / particle.maxLife, 0.7);
        dummy.position.copy(particle.position);
        dummy.rotation.set(particle.rotation.x, particle.rotation.y, particle.rotation.z);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
    }

    return { mesh, burst, update };
  }

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const hitNormal = new THREE.Vector3();
  const hitNormalMatrix = new THREE.Matrix3();
  const lastParticlePoint = new THREE.Vector3(Infinity, Infinity, Infinity);
  const wobbleStates = new Map();
  let raycastMeshes = [];
  let propsRoot = null;
  let standaloneProps = new Set();
  let hoveredProp = null;
  let lastParticleAt = -Infinity;

  function resolveInteractable(object) {
    if (standaloneProps.has(object)) return object;
    let current = object;
    while (current && current.parent !== propsRoot) {
      if (standaloneProps.has(current)) return current;
      current = current.parent;
    }
    return current?.parent === propsRoot ? current : null;
  }

  function wobbleState(object) {
    let state = wobbleStates.get(object);
    if (!state) {
      state = {
        object,
        baseQuaternion: object.quaternion.clone(),
        angleX: 0,
        angleZ: 0,
        velocityX: 0,
        velocityZ: 0,
        phase: Math.random() * Math.PI * 2,
      };
      wobbleStates.set(object, state);
    }
    return state;
  }

  function setHoveredProp(next) {
    if (next === hoveredProp) return;
    hoveredProp = next;
    renderer.domElement.style.cursor = next ? 'pointer' : '';
    if (next) {
      const state = wobbleState(next);
      state.velocityX += pointer.y * 0.22;
      state.velocityZ -= pointer.x * 0.22;
    }
  }

  function updateWobbles(dt, elapsed) {
    const amplitude = THREE.MathUtils.degToRad(INTERACTION.wobbleDegrees);
    const damping = Math.exp(-INTERACTION.damping * dt);
    const delta = new THREE.Quaternion();
    const euler = new THREE.Euler();
    for (const state of wobbleStates.values()) {
      const active = state.object === hoveredProp;
      const targetX = active ? Math.sin(elapsed * 8 + state.phase) * amplitude * 0.65 : 0;
      const targetZ = active ? Math.cos(elapsed * 9 + state.phase) * amplitude : 0;
      state.velocityX = (state.velocityX + (targetX - state.angleX) * INTERACTION.spring * dt) * damping;
      state.velocityZ = (state.velocityZ + (targetZ - state.angleZ) * INTERACTION.spring * dt) * damping;
      state.angleX += state.velocityX * dt;
      state.angleZ += state.velocityZ * dt;
      euler.set(state.angleX, 0, state.angleZ);
      delta.setFromEuler(euler);
      state.object.quaternion.copy(state.baseQuaternion).multiply(delta);
    }
  }

  function handleScenePointer(event) {
    if (!event.isPrimary || raycastMeshes.length === 0) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(raycastMeshes, false)[0];
    if (!hit) {
      setHoveredProp(null);
      return;
    }

    setHoveredProp(resolveInteractable(hit.object));
    const movedEnough = hit.point.distanceToSquared(lastParticlePoint) > 0.012;
    if (event.timeStamp - lastParticleAt >= INTERACTION.particleIntervalMs || movedEnough) {
      hitNormal.copy(hit.face.normal)
        .applyNormalMatrix(hitNormalMatrix.getNormalMatrix(hit.object.matrixWorld)).normalize();
      contactParticles.burst(hit.point, hitNormal);
      lastParticlePoint.copy(hit.point);
      lastParticleAt = event.timeStamp;
    }
  }

  function setupPointerInteraction(stage) {
    propsRoot = stage.getObjectByName('Props');
    if (!propsRoot) throw new Error('scene interaction requires the Props root');
    standaloneProps = new Set(
      ['Eggs', 'tractor_wheel'].map((name) => stage.getObjectByName(name)).filter(Boolean),
    );
    raycastMeshes = [];
    stage.traverse((object) => { if (object.isMesh && object.visible) raycastMeshes.push(object); });
    renderer.domElement.addEventListener('pointermove', handleScenePointer);
    renderer.domElement.addEventListener('pointerdown', handleScenePointer);
    renderer.domElement.addEventListener('pointerleave', () => setHoveredProp(null));
  }

  function segmentIntersectsBox(start, end, box) {
    let minimum = 0;
    let maximum = 1;
    for (const axis of ['x', 'z']) {
      const delta = end[axis] - start[axis];
      if (Math.abs(delta) < 1e-6) {
        if (start[axis] < box.min[axis] || start[axis] > box.max[axis]) return false;
        continue;
      }
      const inverse = 1 / delta;
      let near = (box.min[axis] - start[axis]) * inverse;
      let far = (box.max[axis] - start[axis]) * inverse;
      if (near > far) [near, far] = [far, near];
      minimum = Math.max(minimum, near);
      maximum = Math.min(maximum, far);
      if (minimum > maximum) return false;
    }
    return maximum >= 0 && minimum <= 1;
  }

  function buildRunnerNavigation(stage) {
    const floorRoots = ['floor-active', 'floor-active (1)']
      .map((name) => stage.getObjectByName(name)).filter(Boolean);
    if (floorRoots.length === 0) throw new Error('prop runners require floor roots');

    stage.updateMatrixWorld(true);
    const floorObjects = new Set();
    const floorTiles = [];
    for (const root of floorRoots) {
      root.traverse((object) => {
        floorObjects.add(object);
        if (object.isMesh && /^Floor(?:\b|_)/i.test(object.name)) floorTiles.push(object);
      });
    }
    if (floorTiles.length === 0) throw new Error('prop runners require floor tiles');

    const obstacleBoxes = [];
    stage.traverse((object) => {
      if (!object.isMesh || !object.visible || floorObjects.has(object)) return;
      const box = new THREE.Box3().setFromObject(object);
      if (box.isEmpty() || box.max.y < 0.22) return;
      box.expandByVector(new THREE.Vector3(PROP_RUNNERS.clearance, 0, PROP_RUNNERS.clearance));
      obstacleBoxes.push(box);
    });

    const nodesByPosition = new Map();
    for (const tile of floorTiles) {
      const bounds = new THREE.Box3().setFromObject(tile);
      const position = bounds.getCenter(new THREE.Vector3());
      position.y = bounds.max.y;
      const occupied = obstacleBoxes.some((box) =>
        position.x >= box.min.x && position.x <= box.max.x
        && position.z >= box.min.z && position.z <= box.max.z
        && box.max.y >= position.y + 0.22,
      );
      if (occupied) continue;
      const key = `${Math.round(position.x * 100)}:${Math.round(position.z * 100)}`;
      if (!nodesByPosition.has(key)) nodesByPosition.set(key, { position, neighbors: [] });
    }

    const nodes = [...nodesByPosition.values()];
    const step = PROP_RUNNERS.navigationStep;
    const edgeIsClear = (start, end) => !obstacleBoxes.some((box) => {
      const travelHeight = Math.max(start.y, end.y) + 0.65;
      if (box.max.y < travelHeight || box.min.y > travelHeight + 0.8) return false;
      return segmentIntersectsBox(start, end, box);
    });
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = Math.abs(nodes[i].position.x - nodes[j].position.x);
        const dz = Math.abs(nodes[i].position.z - nodes[j].position.z);
        const adjacent = (Math.abs(dx - step) < 0.2 && dz < 0.2)
          || (Math.abs(dz - step) < 0.2 && dx < 0.2);
        if (!adjacent || !edgeIsClear(nodes[i].position, nodes[j].position)) continue;
        nodes[i].neighbors.push(j);
        nodes[j].neighbors.push(i);
      }
    }

    const visited = new Set();
    let largestComponent = [];
    for (let i = 0; i < nodes.length; i++) {
      if (visited.has(i)) continue;
      const component = [];
      const queue = [i];
      visited.add(i);
      for (let cursor = 0; cursor < queue.length; cursor++) {
        const current = queue[cursor];
        component.push(current);
        for (const neighbor of nodes[current].neighbors) {
          if (visited.has(neighbor)) continue;
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
      if (component.length > largestComponent.length) largestComponent = component;
    }
    if (largestComponent.length < PROP_RUNNERS.count * 3) {
      throw new Error(`prop runner navigation has only ${largestComponent.length} connected nodes`);
    }
    return { nodes, nodeIndices: largestComponent };
  }

  function randomRunnerPath(graph, startIndex, claims, runner) {
    const queue = [startIndex];
    const previous = new Map([[startIndex, -1]]);
    const distance = new Map([[startIndex, 0]]);
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const current = queue[cursor];
      for (const neighbor of graph.nodes[current].neighbors) {
        if (previous.has(neighbor)) continue;
        const owner = claims.get(neighbor);
        if (owner && owner !== runner) continue;
        previous.set(neighbor, current);
        distance.set(neighbor, distance.get(current) + 1);
        queue.push(neighbor);
      }
    }
    const destinations = queue.filter((index) => distance.get(index) >= 3);
    if (destinations.length === 0) return null;
    const target = destinations[Math.floor(Math.random() * destinations.length)];
    const path = [];
    for (let current = target; current !== -1; current = previous.get(current)) path.push(current);
    path.reverse();
    return path;
  }

  function setupPropRunners(stage) {
    const graph = buildRunnerNavigation(stage);
    const actorProfiles = [
      {
        label: 'Cone', pattern: /^Cone$/i, speedScale: 1.15, cadence: 1.2,
        hopScale: 1.3, pitch: 0.035, roll: 0.02, squash: 0.05,
      },
      {
        label: 'Box', pattern: /^Box_Large/i, speedScale: 0.76, cadence: 0.75,
        hopScale: 0.35, pitch: 0.02, roll: 0.09, squash: 0.018,
      },
      {
        label: 'Pack', pattern: /^Pack_(?:1C|2B)/i, speedScale: 1, cadence: 1.45,
        hopScale: 0.8, pitch: 0.07, roll: 0.04, squash: 0.065,
      },
    ];
    if (actorProfiles.length !== PROP_RUNNERS.count) {
      throw new Error('prop runner profile count must match the configured runner count');
    }
    const actorSources = actorProfiles.map((profile) => {
      const candidates = propsRoot.children.filter((object) => profile.pattern.test(object.name));
      if (candidates.length === 0) throw new Error(`prop runners require a ${profile.label} model`);
      return {
        profile,
        source: candidates[Math.floor(Math.random() * candidates.length)],
      };
    });

    const claims = new Map();
    const runners = [];
    const spawnNodes = [...graph.nodeIndices].sort(() => Math.random() - 0.5);
    const worldPosition = new THREE.Vector3();
    const localPosition = new THREE.Vector3();
    const direction = new THREE.Vector3();
    const heading = new THREE.Quaternion();
    const lean = new THREE.Quaternion();
    const motionEuler = new THREE.Euler();
    const up = new THREE.Vector3(0, 1, 0);

    const placeRunner = (runner, position, hop = 0) => {
      localPosition.copy(position);
      localPosition.y += runner.floorOffset + hop;
      stage.worldToLocal(localPosition);
      runner.object.position.copy(localPosition);
    };

    for (let index = 0; index < PROP_RUNNERS.count; index++) {
      const { profile, source } = actorSources[index];
      source.getWorldPosition(worldPosition);
      const sourceBounds = new THREE.Box3().setFromObject(source);
      const object = source.clone(true);
      object.name = `Environment Actor ${profile.label}`;
      stage.add(object);
      const current = spawnNodes[index];
      const runner = {
        object,
        current,
        from: current,
        to: current,
        path: [current],
        moving: false,
        progress: 0,
        speed: THREE.MathUtils.lerp(PROP_RUNNERS.speedMin, PROP_RUNNERS.speedMax, Math.random())
          * profile.speedScale,
        retryAfter: Math.random() * 0.4,
        floorOffset: worldPosition.y - sourceBounds.min.y,
        baseQuaternion: object.quaternion.clone(),
        facingQuaternion: new THREE.Quaternion(),
        baseScale: object.scale.clone(),
        phase: Math.random() * Math.PI * 2,
        profile,
      };
      claims.set(current, runner);
      placeRunner(runner, graph.nodes[current].position);
      runners.push(runner);
    }

    function beginSegment(runner) {
      if (runner.path.length < 2) {
        const path = randomRunnerPath(graph, runner.current, claims, runner);
        if (!path) {
          runner.retryAfter = 0.35 + Math.random() * 0.5;
          return;
        }
        runner.path = path;
      }
      const next = runner.path[1];
      const owner = claims.get(next);
      if (owner && owner !== runner) {
        runner.path = [runner.current];
        runner.retryAfter = 0.2 + Math.random() * 0.35;
        return;
      }
      runner.from = runner.current;
      runner.to = next;
      runner.progress = 0;
      runner.moving = true;
      claims.set(next, runner);
    }

    function update(dt, elapsed) {
      for (const runner of runners) {
        if (!runner.moving) {
          runner.retryAfter -= dt;
          if (runner.retryAfter <= 0) beginSegment(runner);
          if (!runner.moving) {
            placeRunner(runner, graph.nodes[runner.current].position);
            continue;
          }
        }

        worldPosition.lerpVectors(graph.nodes[runner.from].position, graph.nodes[runner.to].position, runner.progress);
        direction.subVectors(graph.nodes[runner.to].position, graph.nodes[runner.from].position).normalize();
        let travel = dt * runner.speed;
        while (runner.moving && travel > 0) {
          const start = graph.nodes[runner.from].position;
          const end = graph.nodes[runner.to].position;
          const distance = start.distanceTo(end);
          const remaining = (1 - runner.progress) * distance;
          const movement = Math.min(travel, remaining);
          travel -= movement;
          runner.phase += movement * Math.PI * runner.profile.cadence;
          direction.subVectors(end, start).normalize();

          if (movement < remaining) {
            runner.progress += movement / distance;
            worldPosition.lerpVectors(start, end, runner.progress);
            break;
          }

          worldPosition.copy(end);
          claims.delete(runner.from);
          runner.current = runner.to;
          runner.path.shift();
          runner.moving = false;
          runner.retryAfter = 0;
          beginSegment(runner);
        }

        const stride = Math.sin(runner.phase);
        placeRunner(
          runner,
          worldPosition,
          Math.abs(stride) * PROP_RUNNERS.hopHeight * runner.profile.hopScale,
        );

        heading.setFromAxisAngle(up, Math.atan2(direction.x, direction.z));
        runner.facingQuaternion.rotateTowards(heading, PROP_RUNNERS.turnSpeed * dt);
        motionEuler.set(
          stride * runner.profile.pitch,
          0,
          Math.cos(runner.phase) * runner.profile.roll,
        );
        lean.setFromEuler(motionEuler);
        runner.object.quaternion.copy(runner.facingQuaternion)
          .multiply(runner.baseQuaternion).multiply(lean);
        const squash = Math.abs(stride) * runner.profile.squash;
        runner.object.scale.set(
          runner.baseScale.x * (1 + squash),
          runner.baseScale.y * (1 - squash),
          runner.baseScale.z * (1 + squash),
        );

      }
    }

    return { update };
  }

  const clock = new THREE.Clock();
  let runnerSystem = null;

  const stage = sceneGltf.scene;
  stage.name = 'Set';
  stage.updateMatrixWorld(true);
  scene.add(stage);
  castAndReceiveShadows(stage);
  prepareMaterials(stage);
  applyObjectEdits(stage);
  setupPointerInteraction(stage);
  runnerSystem = setupPropRunners(stage);

  // The export carries a second, orthographic camera the scene uses elsewhere; the shot is
  // framed by the perspective one.
  const srcCam = sceneGltf.cameras.find((c) => c.isPerspectiveCamera);
  if (!srcCam) throw new Error('the scene GLB has no perspective camera');
  srcCam.updateWorldMatrix(true, false);
  srcCam.getWorldPosition(camBase.pos);
  srcCam.getWorldQuaternion(camBase.quat);
  camBase.fov = srcCam.fov;
  camera.near = srcCam.near;
  camera.far = srcCam.far;
  storyPosition.copy(camBase.pos);
  storyQuaternion.copy(camBase.quat);
  camera.position.copy(camBase.pos);
  camera.quaternion.copy(camBase.quat);
  cameraStoryReady = true;

  const lights = [];
  sceneGltf.scene.traverse((c) => { if (c.isDirectionalLight) lights.push(c); });
  lights.forEach((l, i) => retargetLight(l, i === 0));
  applyLightingPreset(lights);

  let t = 0, elapsed = 0;
  function tick() {
    requestAnimationFrame(tick);
    const dt = Math.min(clock.getDelta(), 0.05);
    elapsed += dt;
    grade.uniforms.uTime.value = elapsed;
    updateWobbles(dt, elapsed);
    if (runnerSystem) runnerSystem.update(dt, elapsed);
    contactParticles.update(dt);

    // dust drift
    const top = DUST_CENTER[1] + DUST_AREA[1] * 0.5;
    const bottom = DUST_CENTER[1] - DUST_AREA[1] * 0.5;
    for (let i = 0; i < DUST_COUNT; i++) {
      const y = i * 3 + 1;
      dust.pos[y] += dt * DUST_RISE * dust.speed[i];
      if (dust.pos[y] > top) dust.pos[y] = bottom;
      dust.pos[i * 3]     += Math.sin(elapsed * 0.4 + dust.phase[i]) * dt * 0.06;
      dust.pos[i * 3 + 2] += Math.cos(elapsed * 0.3 + dust.phase[i]) * dt * 0.04;
    }
    dust.geo.attributes.position.needsUpdate = true;

    t += dt * DRIFT_SPEED;
    updateStoryCamera(dt);
    camera.translateX(Math.sin(t) * DRIFT_X);
    camera.translateY(Math.sin(t * 2) * DRIFT_Y);

    composer.render();
    renderer.autoClear = false;
    renderer.render(particleScene, camera);
    renderer.autoClear = true;
  }

  addEventListener('resize', layout);
  layout();
  tick();
}
