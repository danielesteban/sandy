const Compute = ({ count, size }) => `
struct Projectile {
  position: vec3<f32>,
  direction: vec3<f32>,
  iteration: u32,
  state: u32,
}

@group(0) @binding(0) var<storage, read_write> projectiles : array<Projectile, ${count}>;
@group(0) @binding(1) var<storage, read_write> data : array<atomic<u32>>;

const size : vec3<i32> = vec3<i32>(${size[0]}, ${size[1]}, ${size[2]});

fn getVoxel(pos : vec3<i32>) -> u32 {
  return u32(pos.z * size.x * size.y + pos.y * size.x + pos.x);
}

const neighbors = array<vec3<i32>, 6>(
  vec3<i32>(0, -1, 0),
  vec3<i32>(1, 0, 0),
  vec3<i32>(-1, 0, 0),
  vec3<i32>(0, 0, 1),
  vec3<i32>(0, 0, -1),
  vec3<i32>(0, 1, 0),
);

fn collide(id : u32) {
  let pos : vec3<i32> = vec3<i32>(floor(projectiles[id].position));
  if (any(pos < vec3<i32>(0)) || any(pos >= size)) {
    return;
  }
  if (atomicMin(&data[getVoxel(pos)], 0) != 0) {
    projectiles[id].state = 2;
  }
}

fn detonate(id : u32) {
  let pos : vec3<i32> = vec3<i32>(floor(projectiles[id].position));
  let radius : i32 = 4;
  for (var z : i32 = -radius; z <= radius; z++) {
    for (var y : i32 = -radius; y <= radius; y++) {
      for (var x : i32 = -radius; x <= radius; x++) {
        let npos : vec3<i32> = pos + vec3<i32>(x, y, z);
        if (
          any(npos < vec3<i32>(0))
          || any(npos >= size)
          || length(vec3<f32>(f32(x), f32(y), f32(z))) > (f32(radius) - 0.5)
        ) {
          continue;
        }
        atomicMin(&data[getVoxel(npos)], 0);
      }
    }
  }
}

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  switch (projectiles[id.x].state) {
    default {}
    case 1 {
      collide(id.x);
    }
    case 3 {
      detonate(id.x);
    }
  }
}
`;

class ProjectilesCollide {
  constructor({ count, data, device, projectiles, size }) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: Compute({ count, size }),
        }),
        entryPoint: 'main',
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: projectiles.state },
        },
        {
          binding: 1,
          resource: { buffer: data },
        },
      ],
    });
    this.workgroups = Math.ceil(count / 256);
  }

  compute(pass) {
    const { bindings, pipeline, workgroups } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(workgroups);
  }
}

export default ProjectilesCollide;
