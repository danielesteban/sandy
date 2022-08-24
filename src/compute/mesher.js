const Compute = ({ size }) => `
struct Faces {
  vertexCount : u32,
  instanceCount : atomic<u32>,
  firstVertex : u32,
  firstInstance : u32,
  data : array<f32>
}

@group(0) @binding(0) var<storage, read> data : array<u32>;
@group(0) @binding(1) var<storage, read_write> faces : Faces;

const faceNormals = array<vec3<i32>, 6>(
  vec3<i32>(0, 0, 1),
  vec3<i32>(0, 1, 0),
  vec3<i32>(0, -1, 0),
  vec3<i32>(-1, 0, 0),
  vec3<i32>(1, 0, 0),
  vec3<i32>(0, 0, -1),
);

const size : vec3<i32> = vec3<i32>(${size[0]}, ${size[1]}, ${size[2]});

fn getVoxel(pos : vec3<i32>) -> u32 {
  return u32(pos.z * size.x * size.y + pos.y * size.x + pos.x);
}

fn getValue(pos : vec3<i32>) -> u32 {
  if (pos.y < 0) {
    return 1;
  }
  if (any(pos < vec3<i32>(0)) || any(pos >= size)) {
    return 0;
  }
  return data[getVoxel(pos)];
}

fn pushFace(pos : vec3<i32>, face : i32, value : u32) {
  let offset : u32 = atomicAdd(&faces.instanceCount, 1) * 4;
  faces.data[offset] = f32(pos.x) + 0.5;
  faces.data[offset + 1] = f32(pos.y) + 0.5;
  faces.data[offset + 2] = f32(pos.z) + 0.5;
  faces.data[offset + 3] = f32(i32(value - 1) * 6 + face);
}

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let pos : vec3<i32> = vec3<i32>(id);
  if (any(pos >= size)) {
    return;
  }
  let value = getValue(pos);
  if (value == 0) {
    return;
  }
  for (var face : i32 = 0; face < 6; face++) {
    let npos : vec3<i32> = pos + faceNormals[face];
    if (getValue(npos) == 0) {
      pushFace(pos, face, value);
    }
  }
}
`;

class Mesher {
  constructor({ data, device, faces, size }) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: Compute({ size }),
        }),
        entryPoint: 'main',
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
          resource: { buffer: faces },
        }
      ],
    });
    this.faces = faces;
    this.workgroups = new Uint32Array([
      Math.ceil(size[0] / 64),
      Math.ceil(size[1] / 4),
      size[2],
    ]);
  }

  compute(command) {
    const { bindings, faces, pipeline, workgroups } = this;
    command.clearBuffer(faces, 4, 4);
    const pass = command.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
    pass.end();
  }
}

export default Mesher;
