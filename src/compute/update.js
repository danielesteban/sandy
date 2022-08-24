const Compute = ({ count, size }) => `
@group(0) @binding(0) var<storage, read_write> data : array<u32>;
@group(0) @binding(1) var<uniform> input : array<vec4<i32>, ${count}>;

const size : vec3<i32> = vec3<i32>(${size[0]}, ${size[1]}, ${size[2]});

fn getVoxel(pos : vec3<i32>) -> u32 {
  return u32(pos.z * size.x * size.y + pos.y * size.x + pos.x);
}

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  let i = input[id.x];
  let radius : i32 = 3;
  for (var x : i32 = -radius; x <= radius; x++) {
    for (var z : i32 = -radius; z <= radius; z++) {
      let npos : vec3<i32> = i.xyz + vec3<i32>(x, 0, z);
      if (
        any(npos < vec3<i32>(0))
        || any(npos >= size)
        || length(vec3<f32>(f32(x), 0, f32(z))) > (f32(radius) - 0.5)
      ) {
        continue;
      }
      let voxel = getVoxel(npos);
      if (data[voxel] == 0) {
        data[voxel] = u32(i.w);
      }
    }
  }
}
`;

class Update {
  constructor({ count = 4, data, device, size }) {
    this.input = {
      buffer: device.createBuffer({
        size: count * 4 * Int32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      }),
      data: new Int32Array(count * 4),
    };
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ count, size }),
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
          resource: { buffer: this.input.buffer },
        },
      ],
    });
    this.size = size;

    this.workgroups = Math.ceil(count / 256);
  }

  compute(command) {
    const { bindings, pipeline, workgroups } = this;
    const pass = command.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }
}

export default Update;
