// WebGL2 renderer: camera frame + deformed garment mesh + lighting transfer + occlusion.
const MESH_VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv; in float a_limb; in float a_cut;
uniform vec2 u_size;
out vec2 v_uv; out float v_limb; out float v_cut;
void main() {
  v_uv = a_uv; v_limb = a_limb; v_cut = a_cut;
  vec2 c = a_pos / u_size * 2.0 - 1.0;   // not flipped: row 0 of the FBO = top of the frame
  gl_Position = vec4(c, 0.0, 1.0);
}`;

const MESH_FS = `#version 300 es
precision highp float;
in vec2 v_uv; in float v_limb; in float v_cut;
uniform sampler2D u_tex, u_part;
uniform float u_lodBias, u_group;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_aux;
void main() {
  vec4 c = texture(u_tex, v_uv, u_lodBias);   // premultiplied
  if (v_cut > 0.5) {                          // triangle on a hard body/limb split
    // hard threshold of the bilinearly filtered mask: a smooth boundary, and the two
    // copies stay exactly complementary (no see-through seam in the rest pose)
    float side = step(0.5, texture(u_part, v_uv).r);
    c *= (u_group > 0.5) ? side : 1.0 - side;
  }
  o_col = c;
  o_aux = vec4(v_limb * c.a, 0.0, 0.0, c.a);
}`;

const COMP_VS = `#version 300 es
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  v_uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const COMP_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_cam, u_garm, u_aux, u_seg;
uniform vec2 u_size;
uniform float u_shade, u_illum, u_gain, u_ao, u_lod, u_meanClothes, u_opacity, u_noise, u_time;
uniform vec3 u_wb;
uniform vec2 u_arm[8];       // R: shoulder, elbow, wrist, hand; L: same
uniform float u_armFront[6]; // per arm segment: 1 = in front of the torso (from pose depth)
uniform float u_armR;
uniform int u_hasArms;
uniform vec2 u_shadeP, u_shadeN;  // half-plane where the real clothes say nothing useful
uniform float u_shadeW;
out vec4 fragColor;

vec3 toLin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float segDist(vec2 p, vec2 a, vec2 b) {
  vec2 ab = b - a; float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
  return length(p - a - ab * t);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec2 uv = v_uv;
  vec2 px = uv * u_size;
  vec3 cam = texture(u_cam, uv).rgb;
  vec4 g = texture(u_garm, uv);
  vec4 aux = texture(u_aux, uv);
  vec4 seg = texture(u_seg, uv);
  float clothes = textureLod(u_seg, uv, 2.0).r;
  // only borrow shading from the matching body region (e.g. not from jeans for a top)
  if (u_shadeW > 0.0) clothes *= 1.0 - smoothstep(0.0, u_shadeW, dot(px - u_shadeP, u_shadeN));

  // ---- occlusion: hair/face always in front; arm skin in front of the torso panel
  float limb = aux.a > 0.002 ? aux.r / aux.a : 0.0;
  float armNear = 0.0;
  if (u_hasArms == 1) {
    for (int s = 0; s < 2; s++) {
      for (int j = 0; j < 3; j++) {
        int o = s * 4 + j;
        float d = segDist(px, u_arm[o], u_arm[o + 1]);
        armNear = max(armNear, (1.0 - smoothstep(u_armR, u_armR * 1.7, d)) * u_armFront[s * 3 + j]);
      }
    }
  }
  float occ = max(seg.b, seg.g * (1.0 - limb) * armNear);
  float a = g.a * (1.0 - smoothstep(0.35, 0.75, occ)) * u_opacity;

  // ---- lighting transfer from the real clothes under the garment
  vec3 gc = g.a > 0.002 ? g.rgb / g.a : vec3(0.0);
  vec3 gl = toLin(gc);
  // band-pass: drop fine texture (denim weave, prints, sensor noise), keep folds
  float L  = luma(toLin(textureLod(u_cam, uv, 1.6).rgb));
  float Lb = luma(toLin(textureLod(u_cam, uv, u_lod).rgb));
  float ratio = clamp((L + 0.012) / (Lb + 0.012), 0.6, 1.4);         // folds & wrinkles
  float shade = mix(1.0, ratio, u_shade * clothes);
  float illum = clamp((Lb + 0.01) / (u_meanClothes + 0.01), 0.45, 1.6); // broad light falloff
  illum = mix(1.0, pow(illum, 0.8), u_illum * clothes);
  vec3 lit = gl * shade * illum * u_gain * u_wb;
  lit += (hash(px + u_time) - 0.5) * u_noise;
  vec3 gcol = toSrgb(lit);

  // ---- contact shadow on skin just outside the garment edge
  float gBlur = textureLod(u_garm, uv, 3.5).a;
  float ao = clamp((gBlur - g.a) * 1.6, 0.0, 1.0) * u_ao * seg.a * u_opacity;
  vec3 base = cam * (1.0 - ao);

  fragColor = vec4(mix(base, gcol, a), 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    u[info.name.replace('[0]', '')] = gl.getUniformLocation(p, info.name);
  }
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('瀏覽器不支援 WebGL2');
    this.gl = gl; this.canvas = canvas;
    this.mesh = program(gl, MESH_VS, MESH_FS);
    this.comp = program(gl, COMP_VS, COMP_FS);
    this.camTex = this._tex(true);
    this.segTex = this._tex(true);
    this.garmTex = this._tex(true);
    this.partTex = this._tex(false);
    this.bufCut = gl.createBuffer();
    this.fboCol = this._tex(true); this.fboAux = this._tex(false);
    this.fbo = gl.createFramebuffer();
    this.vao = gl.createVertexArray();
    this.emptyVao = gl.createVertexArray();
    this.bufPos = gl.createBuffer(); this.bufUv = gl.createBuffer(); this.bufLimb = gl.createBuffer();
    this.bufIdxT = gl.createBuffer(); this.bufIdxL = gl.createBuffer();
    this.W = 0; this.H = 0; this.rig = null; this.segVersion = -1;
  }

  _tex(mip) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  resize(W, H) {
    if (W === this.W && H === this.H) return;
    const gl = this.gl;
    this.W = W; this.H = H; this.canvas.width = W; this.canvas.height = H;
    for (const t of [this.fboCol, this.fboAux]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboCol, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.fboAux, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  setGarment(rig) {
    const gl = this.gl;
    this.rig = rig;
    if (!rig) return;
    gl.bindTexture(gl.TEXTURE_2D, this.garmTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, rig.model.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, this.partTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, rig.partW, rig.partH, 0, gl.RED, gl.UNSIGNED_BYTE, rig.part);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    this._uploadGeometry();
  }

  /** (Re)upload vertex buffers + indices; the vertex count can change when the rig re-prepares. */
  _uploadGeometry() {
    const gl = this.gl, rig = this.rig;
    gl.bindVertexArray(this.vao);
    const attr = (buf, name, data, size, usage) => {
      const loc = gl.getAttribLocation(this.mesh.p, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    attr(this.bufPos, 'a_pos', new Float32Array(rig.n * 2), 2, gl.DYNAMIC_DRAW);
    attr(this.bufUv, 'a_uv', rig.uvs, 2, gl.STATIC_DRAW);
    attr(this.bufLimb, 'a_limb', rig.limb, 1, gl.STATIC_DRAW);
    attr(this.bufCut, 'a_cut', rig.cut, 1, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdxT);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, rig.idxTorso, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdxL);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, rig.idxLimb, gl.STATIC_DRAW);
    this.idxType = rig.idxTorso instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    this.indicesVersion = rig.indicesVersion;
  }

  /**
   * video: HTMLVideoElement; seg: {data,w,h,version}; positions: Float32Array or null
   * p: {shade, illum, gain, wb, ao, meanClothes, opacity, arms:[8 points]|null, armR, showGarment}
   */
  render(video, seg, positions, p) {
    const gl = this.gl;
    this.resize(video.videoWidth || video.width, video.videoHeight || video.height);
    const W = this.W, H = this.H;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.generateMipmap(gl.TEXTURE_2D);
    if (seg && seg.version !== this.segVersion) {
      gl.bindTexture(gl.TEXTURE_2D, this.segTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, seg.w, seg.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(seg.data.buffer, seg.data.byteOffset, seg.data.length));
      gl.generateMipmap(gl.TEXTURE_2D);
      this.segVersion = seg.version;
    } else if (!seg && this.segVersion !== -2) {
      gl.bindTexture(gl.TEXTURE_2D, this.segTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
      gl.generateMipmap(gl.TEXTURE_2D);
      this.segVersion = -2;
    }

    // ---- garment pass
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    const drawGarment = this.rig && positions && p.showGarment;
    if (drawGarment) {
      if (this.indicesVersion !== this.rig.indicesVersion) this._uploadGeometry();
      gl.useProgram(this.mesh.p);
      gl.uniform2f(this.mesh.u.u_size, W, H);
      gl.uniform1f(this.mesh.u.u_lodBias, p.lodBias ?? 0.6);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.garmTex);
      gl.uniform1i(this.mesh.u.u_tex, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.partTex);
      gl.uniform1i(this.mesh.u.u_part, 1);
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufPos);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, positions);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform1f(this.mesh.u.u_group, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdxT);
      gl.drawElements(gl.TRIANGLES, this.rig.idxTorso.length, this.idxType, 0);
      gl.uniform1f(this.mesh.u.u_group, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdxL);
      gl.drawElements(gl.TRIANGLES, this.rig.idxLimb.length, this.idxType, 0);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(null);
      gl.bindTexture(gl.TEXTURE_2D, this.fboCol);
      gl.generateMipmap(gl.TEXTURE_2D);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ---- composite
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.comp.p);
    const u = this.comp.u;
    const bind = (unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(u[name], unit); };
    bind(0, this.camTex, 'u_cam'); bind(1, this.fboCol, 'u_garm'); bind(2, this.fboAux, 'u_aux'); bind(3, this.segTex, 'u_seg');
    gl.uniform2f(u.u_size, W, H);
    gl.uniform1f(u.u_shade, p.shade);
    gl.uniform1f(u.u_illum, p.illum);
    gl.uniform1f(u.u_gain, p.gain);
    gl.uniform3fv(u.u_wb, p.wb);
    gl.uniform1f(u.u_ao, p.ao);
    gl.uniform1f(u.u_lod, Math.log2(Math.max(W, H) / 40));
    gl.uniform1f(u.u_meanClothes, p.meanClothes);
    gl.uniform1f(u.u_opacity, drawGarment ? p.opacity : 0);
    gl.uniform1f(u.u_noise, p.noise ?? 0.01);
    const sl = p.shadeLine;
    gl.uniform2f(u.u_shadeP, sl ? sl.p[0] : 0, sl ? sl.p[1] : 0);
    gl.uniform2f(u.u_shadeN, sl ? sl.n[0] : 0, sl ? sl.n[1] : 0);
    gl.uniform1f(u.u_shadeW, sl ? sl.w : 0);
    gl.uniform1f(u.u_time, (performance.now() % 10000) / 10);
    if (p.arms) {
      gl.uniform2fv(u.u_arm, new Float32Array(p.arms.flat()));
      gl.uniform1i(u.u_hasArms, 1);
      gl.uniform1fv(u.u_armFront, new Float32Array(p.armFront || [0, 0, 0, 0, 0, 0]));
      gl.uniform1f(u.u_armR, p.armR);
    } else gl.uniform1i(u.u_hasArms, 0);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
