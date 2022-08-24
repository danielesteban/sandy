import './main.css';
import { vec3 } from 'gl-matrix';
import Camera from './render/camera.js';
import Explosions from './render/explosions.js';
import Input from './render/input.js';
import Ocean from './render/ocean.js';
import Projectiles from './render/projectiles.js';
import Renderer from './render/renderer.js';
import Volume from './compute/volume.js';
import Voxels from './render/voxels.js';

const Main = ({ adapter, device }) => {
  const camera = new Camera({ device });
  const renderer = new Renderer({ adapter, camera, device });
  const volume = new Volume({ device, size: vec3.fromValues(128, 64, 128) });
  document.getElementById('renderer').appendChild(renderer.canvas);
  renderer.setSize(window.innerWidth, window.innerHeight);
  window.addEventListener('resize', () => (
    renderer.setSize(window.innerWidth, window.innerHeight)
  ), false);

  const voxels = new Voxels({
    camera,
    device,
    faces: volume.faces,
    samples: renderer.samples,
  });
  renderer.scene.push(voxels);

  const projectiles = new Projectiles({
    instances: volume.simulation.projectiles.instances,
    camera,
    device,
    samples: renderer.samples,
  });
  renderer.scene.push(projectiles);

  const explosions = new Explosions({
    instances: volume.simulation.explosions.instances,
    camera,
    device,
    geometry: projectiles.geometry,
    samples: renderer.samples,
  });
  renderer.scene.push(explosions);

  const ocean = new Ocean({
    camera,
    device,
    position: vec3.fromValues(64, 6.75, 64),
    samples: renderer.samples,
  });
  renderer.scene.push(ocean);

  let clock = performance.now() / 1000;
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      clock = performance.now() / 1000;
    }
  }, false);

  const input = new Input();
  const center = vec3.fromValues(64, 0, 64);
  vec3.set(camera.target, center[0], 7, center[2]);

  const hues = new Uint32Array([0, 60 * 10, 120 * 10, 180 * 10]);

  const direction = vec3.create();
  const origin = vec3.create();
  const target = vec3.create();

  const animate = () => {
    requestAnimationFrame(animate);

    const time = performance.now() / 1000;
    const delta = Math.min(time - clock, 1);
    clock = time;

    input.update(delta);
    camera.setOrbit(
      input.look.state[0],
      input.look.state[1],
      input.zoom.state * 128
    );

    for (let i = 0; i < 4; i++) {
      const a = time + i * Math.PI * 0.5;
      const d = 32 + Math.sin(time) * 16;
      volume.update.input.data.set([
        Math.floor(center[0] + Math.cos(a) * d),
        63,
        Math.floor(center[2] + Math.sin(a) * d),
        1 + (hues[i]++) % 361,
      ], i * 4);
      device.queue.writeBuffer(volume.update.input.buffer, 0, volume.update.input.data);
    }

    vec3.set(origin, (Math.random() - 0.5) * 128, 64, (Math.random() - 0.5) * 128);
    vec3.add(origin, origin, center);
    vec3.set(target, (Math.random() - 0.5) * 112, 0, (Math.random() - 0.5) * 112);
    vec3.add(target, target, center);
    vec3.sub(direction, target, origin);
    vec3.normalize(direction, direction);
    volume.simulation.shoot(direction, origin);

    const command = device.createCommandEncoder();
    volume.update.compute(command);
    volume.compute(command, delta);
    renderer.render(command);
    device.queue.submit([command.finish()]);
  };

  requestAnimationFrame(animate);
};

const GPU = async () => {
  if (!navigator.gpu) {
    throw new Error('WebGPU support');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU adapter');
  }
  const device = await adapter.requestDevice();
  const check = device.createShaderModule({
    code: `const checkConstSupport : f32 = 1;`,
  });
  const { messages } = await check.compilationInfo();
  if (messages.find(({ type }) => type === 'error')) {
    throw new Error('WGSL const support');
  }
  return { adapter, device };
};

GPU()
  .then(Main)
  .catch((e) => {
    console.error(e);
    document.getElementById('canary').classList.add('enabled');
  })
  .finally(() => document.getElementById('loading').classList.remove('enabled'));
