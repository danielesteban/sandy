const Vertex = `
struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) instance : vec4<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) depth : f32,
}

struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : Camera;

const faces = array<mat3x3<f32>, 6>(
  mat3x3<f32>(vec3<f32>(1, 0, 0), vec3<f32>(0, 1, 0), vec3<f32>(0, 0, 1)),
  mat3x3<f32>(vec3<f32>(1, 0, 0), vec3<f32>(0, 0, -1), vec3<f32>(0, 1, 0)),
  mat3x3<f32>(vec3<f32>(1, 0, 0), vec3<f32>(0, 0, 1), vec3<f32>(0, -1, 0)),
  mat3x3<f32>(vec3<f32>(0, 0, 1), vec3<f32>(0, 1, 0), vec3<f32>(-1, 0, 0)),
  mat3x3<f32>(vec3<f32>(0, 0, -1), vec3<f32>(0, 1, 0), vec3<f32>(1, 0, 0)),
  mat3x3<f32>(vec3<f32>(-1, 0, 0), vec3<f32>(0, 1, 0), vec3<f32>(0, 0, -1)),
);

fn hue2Rgb(p : f32, q : f32, t : f32) -> f32 {
  var h : f32 = t;
  if (h < 0) { h += 1; }
  if (h > 1) { h -= 1; }
  if (h < 1 / 6.0) { return p + (q - p) * 6 * h; }
  if (h < 1 / 2.0) { return q; }
  if (h < 2 / 3.0) { return p + (q - p) * (2.0 / 3.0 - h) * 6; }
  return p;
}

fn hsl2Rgb(h : f32, s: f32, l: f32) -> vec3<f32> {
  var rgb : vec3<f32> = vec3<f32>(0, 0, 0);
  var q : f32;
  if (l < 0.5) {
    q = l * (1 + s);
  } else {
    q = l + s - l * s;
  }
  var p : f32 = 2 * l - q;
  rgb.r = hue2Rgb(p, q, h + 1 / 3.0);
  rgb.g = hue2Rgb(p, q, h);
  rgb.b = hue2Rgb(p, q, h - 1 / 3.0);
  return rgb;
}

@vertex
fn main(voxel : VertexInput) -> VertexOutput {
  let model : mat3x3<f32> = faces[i32(voxel.instance.w % 6)];
  let position : vec3<f32> = model * voxel.position + voxel.instance.xyz;
  let mvPosition : vec4<f32> = camera.view * vec4<f32>(position, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  let s = max(smoothstep(0, 6, position.y), 0.1);
  out.color = hsl2Rgb(floor(voxel.instance.w / 6) / 360, 0.4, 0.4);
  if (s < 1) {
    out.color = mix(vec3<f32>(0.1, 0.2, 0.3) * s, out.color, s);
  }
  out.normal = model[2];
  out.depth = -mvPosition.z;
  return out;
}
`;

const Fragment = `
struct FragmentInput {
  @location(0) color : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) depth : f32,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
  @location(1) data : vec4<f32>,
}

@fragment
fn main(face : FragmentInput) -> FragmentOutput {
  var output : FragmentOutput;
  output.color = vec4<f32>(face.color, 1);
  output.data = vec4<f32>(normalize(face.normal), face.depth);
  return output;
}
`;

const Face = (device) => {
  const buffer = device.createBuffer({
    mappedAtCreation: true,
    size: 18 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5, -0.5,  0.5,
  ]);
  buffer.unmap();
  return buffer;
};

class Voxels {
  constructor({ camera, device, faces, samples }) {
    this.device = device;
    this.faces = faces;
    this.geometry = Face(device);
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        buffers: [
          {
            arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
              },
            ],
          },
          {
            arrayStride: 4 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: 'instance',
            attributes: [
              {
                shaderLocation: 1,
                offset: 0,
                format: 'float32x4',
              },
            ],
          }
        ],
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Vertex,
        }),
      },
      fragment: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Fragment,
        }),
        targets: [
          { format: 'rgba16float' },
          { format: 'rgba16float' },
        ],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
      multisample: {
        count: samples,
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: camera.buffer },
        },
      ],
    });
  }

  render(pass) {
    const { bindings, faces, geometry, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    pass.setVertexBuffer(1, faces, 16);
    pass.drawIndirect(faces, 0);
  }
}

export default Voxels;
