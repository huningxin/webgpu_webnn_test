import glslangModule from 'https://unpkg.com/@webgpu/glslang/web/glslang.js';

const TENSOR_DIMS = [2, 2];
const TENSOR_SIZE = 4;

const nn = navigator.ml.getNeuralNetworkContext();

async function createNNModel(device) {
  let operandIndex = 0;
  const TENSOR_DATA = [1, 1, 1, 1];
  const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMS};
  const scalarInt32Type = {type: nn.INT32};

  const model = await nn.createModel();

  const fusedActivationFuncNone = operandIndex++;
  model.addOperand(scalarInt32Type);
  model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

  const constant = operandIndex++;
  model.addOperand(float32TensorType);
  model.setOperandValue(constant, new Float32Array(TENSOR_DATA));

  const input = operandIndex++;
  model.addOperand(float32TensorType);

  const output = operandIndex++;
  model.addOperand(float32TensorType);

  model.addOperation(nn.ADD, [input, constant, fusedActivationFuncNone], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();

  return model;
}

let additionWebGPUCode = null;
async function runMatrixAddByWebGPU(device, gpuBufferSecondMatrix) {
  // First Matrix
  const firstMatrix = new Float32Array([1, 2, 3, 4]);

  const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = await device.createBufferMappedAsync({
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  
  // Result Matrix

  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (TENSOR_SIZE);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Shape Matrix

  const shape = new Float32Array(TENSOR_DIMS);

  const [gpuBufferShape, arrayBufferShape] = await device.createBufferMappedAsync({
    size: shape.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferShape).set(shape);
  gpuBufferShape.unmap();

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    bindings: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      }
    ]
  });


  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    bindings: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferShape
        }
      }
    ]
  });
  

  // Compute shader code (GLSL)

  const computeShaderCode = `#version 450

  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
  } resultMatrix;

  layout(std430, set = 0, binding = 3) buffer Shape {
    vec2 size;
  } shape;

  void main() {
    uint index = gl_GlobalInvocationID.y + gl_GlobalInvocationID.x * int(shape.size.x);
    resultMatrix.numbers[index] = firstMatrix.numbers[index] + secondMatrix.numbers[index];
  }
  `;



  // Pipeline setup

  if (additionWebGPUCode == null) {
    const glslang = await glslangModule();
    additionWebGPUCode = glslang.compileGLSL(computeShaderCode, "compute");
  }
  
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: additionWebGPUCode
      }),
      entryPoint: "main"
    }
  });


  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(shape[0], shape[0]);
  passEncoder.endPass();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultMatrixBufferSize /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.getQueue().submit([gpuCommands]);


  // Read buffer.
  const arrayBuffer = await gpuReadBuffer.mapReadAsync();

  const resultArray = new Float32Array(arrayBuffer);

  //console.log(resultArray);
  return resultArray;
}

let multiplyWebGPUCode = null;
async function runMatrixMultiplyByWebGPU(device) {
  // First Matrix
  const rows = 2;
  const columns = 4;
  const firstMatrix = new Float32Array([
    1, 2, 3, 4,
    5, 6, 7, 8
  ]);

  const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = await device.createBufferMappedAsync({
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();


  // Second Matrix

  const secondMatrix = new Float32Array([
    1, 2,
    3, 4,
    5, 6,
    7, 8
  ]);

  const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = await device.createBufferMappedAsync({
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  
  // Result Matrix
  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (rows * rows);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Shape Matrix

  const shape = new Float32Array([rows, columns]);

  const [gpuBufferShape, arrayBufferShape] = await device.createBufferMappedAsync({
    size: shape.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferShape).set(shape);
  gpuBufferShape.unmap();

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    bindings: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      }
    ]
  });


  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    bindings: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferShape
        }
      }
    ]
  });
  

  // Compute shader code (GLSL)

  const computeShaderCode = `#version 450

  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
  } resultMatrix;

  layout(std430, set = 0, binding = 3) buffer Shape {
    vec2 size;
  } shape;

  void main() {
    ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    float result = 0.0;
    for (int i = 0; i < shape.size.y; i++) {
      int a = i + resultCell.x * int(shape.size.y);
      int b = resultCell.y + i * int(shape.size.x);
      result += firstMatrix.numbers[a] * secondMatrix.numbers[b];
    }

    int index = resultCell.y + resultCell.x * int(shape.size.x);
    resultMatrix.numbers[index] = result;
  }
  `;



  // Pipeline setup

  if (multiplyWebGPUCode == null) {
    const glslang = await glslangModule();
    multiplyWebGPUCode = glslang.compileGLSL(computeShaderCode, "compute");
  }
  
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: multiplyWebGPUCode
      }),
      entryPoint: "main"
    }
  });


  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(shape[0], shape[0]);
  passEncoder.endPass();

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.getQueue().submit([gpuCommands]);

  return resultMatrixBuffer;
}

(async () => {
  
  if (!navigator.gpu) {
    console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Create nn graph
  const model = await createNNModel();

  const compilation = await model.createCompilation();
  compilation.setGPUDevice(device);
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  //    WebGPU multiply
  //          |
  //   inputGPUBuffer (result of multiplication)
  //          |
  //     WebNN addition
  //          |
  //   outputGPUBuffer (result of addition)
  //          |
  //     WebGPU addition  

  const count = 100;
  let startTime = performance.now();
  for (let i = 0; i < count; i++) {
    if (i == 1) {
      let warmupTime = performance.now() - startTime;
      console.log(`warmup time ${warmupTime}`);
      startTime = performance.now();
    }
    const outputGPUBufferSize = Float32Array.BYTES_PER_ELEMENT * (TENSOR_SIZE);
    const outputGPUBuffer = device.createBuffer({
      size: outputGPUBufferSize,
      usage: GPUBufferUsage.STORAGE
    });

    const inputGPUBuffer = await runMatrixMultiplyByWebGPU(device);

    // Execute nn graph by GPU command buffer
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputGPUBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputGPUBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    const gpuCommands = commandEncoder.finish();
    device.getQueue().submit([gpuCommands]);

    let result = await runMatrixAddByWebGPU(device, outputGPUBuffer);
    if (i == count - 1) {
      console.log(`result: [${result}]`);
    }
  }
  let meanTime = (performance.now() - startTime)/(count - 1);
  console.log(`mean time ${meanTime.toFixed(2)} ms`);

})();
