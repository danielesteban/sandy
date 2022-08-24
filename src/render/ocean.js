const Vertex = ({ position }) => `
struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) origin : vec2<f32>,
  @location(2) light : f32,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) depth : f32,
  @location(1) fog : f32,
  @location(2) light : f32,
}

struct Camera {
  projection : mat4x4<f32>,
  view : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera : Camera;

const origin : vec3<f32> = vec3<f32>(${position[0]}, ${position[1]}, ${position[2]});

@vertex
fn main(vertex : VertexInput) -> VertexOutput {
  let position : vec3<f32> = vertex.position + vec3<f32>(vertex.origin.x, 0, vertex.origin.y);
  let mvPosition : vec4<f32> = camera.view * vec4<f32>(position + origin, 1);
  var out : VertexOutput;
  out.position = camera.projection * mvPosition;
  out.depth = -mvPosition.z;
  out.fog = length(position);
  out.light = 0.75 + vertex.light * 0.25;
  return out;
}
`;

const Fragment = `
struct FragmentInput {
  @location(0) depth : f32,
  @location(1) fog : f32,
  @location(2) light : f32,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
}

const fogDensity : f32 = 0.005;

@fragment
fn main(face : FragmentInput) -> FragmentOutput {
  var output : FragmentOutput;
  output.color = vec4<f32>(vec3<f32>(0.1, 0.2, 0.3) * face.light, exp(-fogDensity * fogDensity * face.fog * face.fog) * 0.7);
  return output;
}
`;

const Plane = (device) => {
  const buffer = device.createBuffer({
    mappedAtCreation: true,
    size: 18 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
  });
  new Float32Array(buffer.getMappedRange()).set([
    -0.5, 0, 0.5,
     0.5, 0, 0.5,
     0.5, 0, -0.5,
     0.5, 0, -0.5,
    -0.5, 0, -0.5,
    -0.5, 0, 0.5,
  ]);
  buffer.unmap();
  return buffer;
};

const Instances = (device) => {
  const buffer = device.createBuffer({
    mappedAtCreation: true,
    size: 1201 * 1201 * 3 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX,
  });
  const data = new Float32Array(buffer.getMappedRange());
  for (let i = 0, z = -600; z <= 600; z++) {
    for (let x = -600; x <= 600; x++, i += 3) {
      data.set([x + 0.5, z + 0.5, Math.random()], i);
    }
  }
  buffer.unmap();
  return buffer;
};

class Ocean {
  constructor({ camera, device, position, samples }) {
    this.device = device;
    this.geometry = Plane(device);
    this.instances = Instances(device);
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
            arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: 'instance',
            attributes: [
              {
                shaderLocation: 1,
                offset: 0,
                format: 'float32x2',
              },
              {
                shaderLocation: 2,
                offset: 2 * Float32Array.BYTES_PER_ELEMENT,
                format: 'float32',
              },
            ],
          }
        ],
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Vertex({ position }),
        }),
      },
      fragment: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Fragment,
        }),
        targets: [
          {
            format: 'rgba16float',
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
          { format: 'rgba16float', writeMask: 0 },
        ],
      },
      primitive: {
        topology: 'triangle-list',
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
    const { bindings, geometry, instances, pipeline } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.setVertexBuffer(0, geometry);
    pass.setVertexBuffer(1, instances);
    pass.draw(6, 1201 * 1201, 0, 0);
  }
}

export default Ocean;
