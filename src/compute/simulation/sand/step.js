const Compute = ({ size }) => `
struct Uniforms {
  offset : atomic<u32>,
  y : i32,
}

@group(0) @binding(0) var<storage, read_write> data : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> uniforms : Uniforms;

const size : vec3<i32> = vec3<i32>(${size[0]}, ${size[1]}, ${size[2]});

fn getVoxel(pos : vec3<i32>) -> u32 {
  return u32(pos.z * size.x * size.y + pos.y * size.x + pos.x);
}

const neighbors = array<vec3<i32>, 5>(
  vec3<i32>(0, -1, 0),
  vec3<i32>(0, -1, -1),
  vec3<i32>(-1, -1, 0),
  vec3<i32>(0, -1, 1),
  vec3<i32>(1, -1, 0),
);

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(i32(id.x), uniforms.y + 1, i32(id.y));
  if (any(pos >= size)) {
    return;
  }
  let voxel = getVoxel(pos);
  let value = atomicLoad(&data[voxel]);
  if (value == 0) {
    return;
  }
  let o : u32 = atomicAdd(&uniforms.offset, 1);
  for (var n : u32 = 0; n < 5; n++) {
    var ni : u32 = 0;
    if (n > 0) {
      ni = 1 + ((n + o) % 4);
    }
    let npos = pos + neighbors[ni];
    if (any(npos < vec3<i32>(0)) || any(npos >= size)) {
      continue;
    }
    let nvoxel = getVoxel(npos);
    if (atomicCompareExchangeWeak(&data[nvoxel], 0, value).exchanged) {
      atomicStore(&data[voxel], 0);
      break;
    }
  }
}
`;

class SandStep {
  constructor({ data, device, size, uniforms }) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ size }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: data },
        },
        {
          binding: 1,
          resource: { buffer: uniforms },
        },
      ],
    });
    this.workgroups = new Uint32Array([
      Math.ceil(size[0] / 64),
      Math.ceil(size[2] / 4),
    ]);
  }

  compute(pass) {
    const { bindings, pipeline, workgroups } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1]);
  }
}

export default SandStep;
