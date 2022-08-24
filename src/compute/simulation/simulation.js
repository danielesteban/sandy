import { vec3 } from 'gl-matrix';
import ExplosionsMesh from './explosions/mesh.js';
import ExplosionsStep from './explosions/step.js';
import ProjectilesCollide from './projectiles/collide.js';
import ProjectilesStep from './projectiles/step.js';
import SandSetup from './sand/setup.js';
import SandStep from './sand/step.js';

class Simulation {
  constructor({ count = 128, data, device, size }) {
    this.device = device;
    {
      const data = new Float32Array(1);
      this.delta = {
        buffer: device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
        }),
        data,
      };
    }
    {
      const instancesPerMesh = 64;
      const instances = device.createBuffer({
        mappedAtCreation: true,
        size: (
          5 * Uint32Array.BYTES_PER_ELEMENT
          + count * instancesPerMesh * 4 * Float32Array.BYTES_PER_ELEMENT
        ),
        usage: (
          GPUBufferUsage.COPY_DST
          | GPUBufferUsage.INDIRECT
          | GPUBufferUsage.STORAGE
          | GPUBufferUsage.VERTEX
        ),
      });
      new Uint32Array(instances.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0] = 36;
      instances.unmap();
      const meshes = device.createBuffer({
        size: count * 4 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
      });
      const state = device.createBuffer({
        size: count * 8 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
      });
      const workgroups = device.createBuffer({
        mappedAtCreation: true,
        size: 3 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
      });
      new Uint32Array(workgroups.getMappedRange()).set([1, 0, 1]);
      workgroups.unmap();
      this.explosions = {
        instances,
        instancesPerMesh,
        meshes,
        state,
        workgroups,
      };
    }
    {
      const instances = device.createBuffer({
        mappedAtCreation: true,
        size: (
          5 * Uint32Array.BYTES_PER_ELEMENT
          + count * 6 * Float32Array.BYTES_PER_ELEMENT
        ),
        usage: (
          GPUBufferUsage.COPY_DST
          | GPUBufferUsage.INDIRECT
          | GPUBufferUsage.STORAGE
          | GPUBufferUsage.VERTEX
        ),
      });
      new Uint32Array(instances.getMappedRange(0, Uint32Array.BYTES_PER_ELEMENT))[0] = 36;
      instances.unmap();
      const state = device.createBuffer({
        size: count * 12 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
      });
      this.projectiles = {
        instances,
        state,
      };
    }
    {
      const uniforms = device.createBuffer({
        mappedAtCreation: true,
        size: 2 * Int32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
      });
      new Int32Array(uniforms.getMappedRange())[1] = size[1] - 2; 
      uniforms.unmap();
      this.sand = {
        uniforms,
      };
    }
    this.pipelines = {
      explosions: {
        mesh: new ExplosionsMesh({
          count,
          device,
          explosions: this.explosions,
        }),
        step: new ExplosionsStep({
          count,
          delta: this.delta,
          device,
          explosions: this.explosions,
          projectiles: this.projectiles,
        }),
      },
      projectiles: {
        collide: new ProjectilesCollide({
          count,
          data,
          device,
          projectiles: this.projectiles,
          size,
        }),
        step: new ProjectilesStep({
          count,
          delta: this.delta,
          device,
          explosions: this.explosions,
          projectiles: this.projectiles,
        }),
      },
      sand: {
        setup: new SandSetup({ device, size, uniforms: this.sand.uniforms }),
        step: new SandStep({ data, device, size, uniforms: this.sand.uniforms }),
      },
    };
    this.size = size;
  }

  compute(command, delta) {
    const { delta: { buffer, data }, device, pipelines: { explosions, projectiles, sand }, size } = this;
    data[0] = delta;
    device.queue.writeBuffer(buffer, 0, data);
    command.clearBuffer(this.explosions.instances, 4, 4);
    command.clearBuffer(this.explosions.workgroups, 4, 4);
    command.clearBuffer(this.projectiles.instances, 4, 4);
    const pass = command.beginComputePass();
    for (let y = 0; y < (size[1] - 1); y++) {
      sand.setup.compute(pass);
      sand.step.compute(pass);
    }
    explosions.step.compute(pass);
    explosions.mesh.compute(pass);
    projectiles.step.compute(pass);
    projectiles.collide.compute(pass);
    pass.end();
  }

  shoot(direction, origin) {
    const { device, pipelines: { projectiles: { step: { input } } } } = this;
    vec3.copy(input.position, origin);
    vec3.copy(input.direction, direction);
    input.enabled[0] = 1;
    device.queue.writeBuffer(input.buffer, 0, input.data);
  }
}

export default Simulation;
