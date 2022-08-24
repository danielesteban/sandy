import Mesher from './mesher.js';
import Simulation from './simulation/simulation.js';
import Update from './update.js';

class Volume {
  constructor({ device, size }) {
    this.data = device.createBuffer({
      size: size[0] * size[1] * size[2] * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });

    this.faces = device.createBuffer({
      size: (
        4 * Uint32Array.BYTES_PER_ELEMENT
        + Math.ceil(size[0] * size[1] * size[2] * 0.5) * 4 * Float32Array.BYTES_PER_ELEMENT
      ),
      usage: (
        GPUBufferUsage.COPY_DST
        | GPUBufferUsage.INDIRECT
        | GPUBufferUsage.STORAGE
        | GPUBufferUsage.VERTEX
      ),
    });
    device.queue.writeBuffer(this.faces, 0, new Uint32Array([6]));

    this.mesher = new Mesher({ data: this.data, device, faces: this.faces, size });
    this.simulation = new Simulation({ data: this.data, device, size });
    this.update = new Update({ data: this.data, device, size });
  }

  compute(command, delta) {
    const { mesher, simulation } = this;
    simulation.compute(command, delta);
    mesher.compute(command);
  }
}

export default Volume;
