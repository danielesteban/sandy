const Compute = ({ count }) => `
struct Input {
  position: vec3<f32>,
  direction: vec3<f32>,
  enabled: atomic<u32>,
}

struct Instances {
  indexCount : u32,
  instanceCount : atomic<u32>,
  firstIndex : u32,
  baseVertex : u32,
  firstInstance : u32,
  data : array<f32, ${count * 6}>,
}

struct Projectile {
  position: vec3<f32>,
  direction: vec3<f32>,
  iteration: u32,
  state: u32,
}


@group(0) @binding(0) var<uniform> delta : f32;
@group(0) @binding(1) var<storage, read_write> input : Input;
@group(0) @binding(2) var<storage, read_write> instances : Instances;
@group(0) @binding(3) var<storage, read_write> projectiles : array<Projectile, ${count}>;

fn instanceProjectile(position : vec3<f32>, direction : vec3<f32>) {
  let offset : u32 = atomicAdd(&instances.instanceCount, 1) * 6;
  instances.data[offset] = position.x;
  instances.data[offset + 1] = position.y;
  instances.data[offset + 2] = position.z;
  instances.data[offset + 3] = direction.x;
  instances.data[offset + 4] = direction.y;
  instances.data[offset + 5] = direction.z;
}

@compute @workgroup_size(${Math.min(count, 256)})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x >= ${count}) {
    return;
  }
  switch (projectiles[id.x].state) {
    default {}
    case 1 {
      projectiles[id.x].iteration++;
      if (projectiles[id.x].iteration > 128) {
        projectiles[id.x].state = 0;
        return;
      }
      let direction : vec3<f32> = projectiles[id.x].direction;
      projectiles[id.x].position += direction * delta * 60;
      instanceProjectile(projectiles[id.x].position, direction);
      if (projectiles[id.x].position.y <= 0) {
        projectiles[id.x].state = 2;
      }
      return;
    }
    case 2 {
      projectiles[id.x].state = 3;
      return;
    }
    case 3 {
      projectiles[id.x].state = 0;
    }
  }
  if (atomicMin(&input.enabled, 0) != 0) {
    projectiles[id.x].position = input.position; 
    projectiles[id.x].direction = input.direction; 
    projectiles[id.x].iteration = 0;
    projectiles[id.x].state = 1;
  }
}
`;

class ProjectilesStep {
  constructor({ count, delta, device, projectiles }) {
    {
      const data = new Float32Array(8);
      this.input = {
        buffer: device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        }),
        data: data.buffer,
        position: data.subarray(0, 3),
        direction: data.subarray(4, 7),
        enabled: new Uint32Array(data.buffer, 28, 1),
      };
    }
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: Compute({ count }),
        }),
      },
    });
    this.bindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: delta.buffer },
        },
        {
          binding: 1,
          resource: { buffer: this.input.buffer },
        },
        {
          binding: 2,
          resource: { buffer: projectiles.instances },
        },
        {
          binding: 3,
          resource: { buffer: projectiles.state },
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

export default ProjectilesStep;
