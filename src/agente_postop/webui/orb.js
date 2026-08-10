// Orbe 3D — el rostro del agente. Respira en reposo, se agita con la voz del paciente
// mientras escucha, titila suavemente mientras "piensa", y ondula con su propia voz
// mientras responde. Sin bloom real (evitamos el postprocessing de three.js por peso);
// el brillo se simula con capas de esferas translúcidas superpuestas.

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

// Ruido simplex 3D (Ashima Arts / Stefan Gustavson, dominio público / MIT — implementación
// estándar usada en incontables shaders de three.js).
const SIMPLEX_GLSL = `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);
  const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));
  vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);
  vec3 l=1.0-g;
  vec3 i1=min(g.xyz,l.zxy);
  vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;
  vec3 x2=x0-i2+C.yyy;
  vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;
  vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z);
  vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;
  vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);
  vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;
  vec4 s1=floor(b1)*2.0+1.0;
  vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
  vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);
  vec3 p1=vec3(a0.zw,h.y);
  vec3 p2=vec3(a1.xy,h.z);
  vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
  m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
`;

const VERTEX_SHADER = `
  uniform float uTime;
  uniform float uAmplitude;
  uniform float uFrequency;
  varying float vDisplacement;
  varying vec3 vNormal;
  ${SIMPLEX_GLSL}
  void main() {
    vNormal = normal;
    float noise = snoise(position * uFrequency + uTime * 0.35);
    float displacement = noise * uAmplitude;
    vDisplacement = displacement;
    vec3 newPosition = position + normal * displacement;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  uniform vec3 uColor;
  uniform vec3 uColorDeep;
  varying float vDisplacement;
  varying vec3 vNormal;
  void main() {
    float fresnel = pow(1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0))), 2.2);
    vec3 base = mix(uColorDeep, uColor, clamp(vDisplacement * 2.0 + 0.5, 0.0, 1.0));
    vec3 color = mix(base, vec3(1.0), fresnel * 0.35);
    gl_FragColor = vec4(color, 1.0);
  }
`;

// Shader de halo — fresnel puro: opaco en el borde (silueta), transparente de frente.
// Es la técnica estándar para un "glow" atmosférico creíble sin postprocessing de bloom.
const HALO_VERTEX = `
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    gl_Position = projectionMatrix * mvPosition;
  }
`;
const HALO_FRAGMENT = `
  uniform vec3 uColor;
  uniform float uPower;
  uniform float uIntensidad;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  void main() {
    vec3 viewDir = normalize(vViewPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, normalize(vNormal)), 0.0), uPower);
    gl_FragColor = vec4(uColor, fresnel * uIntensidad);
  }
`;

const PALETAS = {
  idle: { color: 0x9fd6b8, deep: 0x3f6b56 },
  listening: { color: 0x8fd0c9, deep: 0x2f6b66 },
  thinking: { color: 0xdcc27a, deep: 0x7a5f24 },
  speaking: { color: 0x9fd6b8, deep: 0x3f6b56 },
  rojo: { color: 0xe0785f, deep: 0x8a3a24 },
};

export class Orb {
  constructor(container) {
    this.container = container;
    this.state = "idle";
    this.amplitude = 0.0;
    this.targetAmplitude = 0.0;
    this.clock = new THREE.Clock();

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.z = 3.4;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.uniforms = {
      uTime: { value: 0 },
      uAmplitude: { value: 0.08 },
      uFrequency: { value: 1.6 },
      uColor: { value: new THREE.Color(PALETAS.idle.color) },
      uColorDeep: { value: new THREE.Color(PALETAS.idle.deep) },
    };

    const geometry = new THREE.IcosahedronGeometry(1, 24);
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: this.uniforms,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);

    // Halo de brillo — capas concéntricas con shader de fresnel (opaco en la silueta,
    // transparente de frente), additive blending. Simula un glow atmosférico real.
    this.halos = [];
    for (const [scale, power, intensidad] of [[1.4, 1.1, 0.85], [1.7, 1.4, 0.5], [2.1, 1.8, 0.24]]) {
      const halo = new THREE.Mesh(
        new THREE.IcosahedronGeometry(1, 16),
        new THREE.ShaderMaterial({
          vertexShader: HALO_VERTEX,
          fragmentShader: HALO_FRAGMENT,
          uniforms: {
            uColor: { value: new THREE.Color(PALETAS.idle.color) },
            uPower: { value: power },
            uIntensidad: { value: intensidad },
          },
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.FrontSide,
        })
      );
      halo.scale.setScalar(scale);
      halo.userData.baseScale = scale;
      this.scene.add(halo);
      this.halos.push(halo);
    }

    this._resize();
    window.addEventListener("resize", () => this._resize());
    this._animate();
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setState(state) {
    if (!(state in PALETAS)) return;
    this.state = state;
    const paleta = PALETAS[state];
    this._colorObjetivo = new THREE.Color(paleta.color);
    this._colorDeepObjetivo = new THREE.Color(paleta.deep);
    this._haloColorObjetivo = new THREE.Color(paleta.color);
  }

  setAmplitude(value) {
    this.targetAmplitude = Math.min(1, Math.max(0, value));
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const t = this.clock.getElapsedTime();
    this.uniforms.uTime.value = t;

    this.amplitude += (this.targetAmplitude - this.amplitude) * 0.15;
    const base = this.state === "thinking" ? 0.09 + Math.sin(t * 3.2) * 0.03 : 0.07;
    this.uniforms.uAmplitude.value = base + this.amplitude * 0.22;

    if (this._colorObjetivo) {
      this.uniforms.uColor.value.lerp(this._colorObjetivo, 0.06);
      this.uniforms.uColorDeep.value.lerp(this._colorDeepObjetivo, 0.06);
    }

    const pulseScale = 1 + Math.sin(t * (this.state === "thinking" ? 2.4 : 0.8)) * 0.02 + this.amplitude * 0.06;
    this.mesh.scale.setScalar(pulseScale);
    this.mesh.rotation.y = t * 0.12;
    this.mesh.rotation.x = Math.sin(t * 0.15) * 0.1;

    this.halos.forEach((halo, i) => {
      halo.scale.setScalar(halo.userData.baseScale * (1 + this.amplitude * 0.1));
      halo.rotation.y = -t * (0.05 + i * 0.02);
      if (this._haloColorObjetivo) {
        halo.material.uniforms.uColor.value.lerp(this._haloColorObjetivo, 0.06);
      }
    });

    this.renderer.render(this.scene, this.camera);
  }
}
