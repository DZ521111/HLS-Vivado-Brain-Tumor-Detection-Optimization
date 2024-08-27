# HLS-Vivado-Brain-Tumor-Detection-Optimization
Optimized a brain tumor detection model, initially running it through the HLS4ML tool to achieve a latency range of 367,441 to 3,458,344 cycles. Converted the design to C and manually optimized it in Vivado, successfully reducing the latency to 1,249,664 cycles, approaching the performance of the HLS4ML-optimized model.

## Project Overview

This project focuses on optimizing a brain tumor detection model using High-Level Synthesis (HLS) tools. The initial model was run through the HLS4ML tool, and further optimizations were carried out manually using Vivado HLS to improve latency and performance.

## Project Details

- **Duration**: July 2023 - January 2024
- **Supervisor**: Dr. Chandan Karfa, Associate Professor, Dept. of CSE, IIT Guwahati
- **Tools Used**: HLS4ML, Vivado HLS, Vivado

## Objectives

1. To optimize the latency of a brain tumor detection model.
2. To compare performance between the HLS4ML-optimized model and manually optimized models using Vivado.
3. To analyze the impact of different optimization techniques on hardware utilization and latency.

## Performance Summary

### HLS4ML Optimization Results

- **Latency**: 367,441 to 3,458,344 clock cycles
- **Estimated Timing**: 5.435 ns
- **Clock Uncertainty**: 0.62 ns
- **Pipeline Type**: Dataflow

### Vivado HLS Optimization Results

- **Minimum Latency**: 1,249,664 clock cycles (reduced from HLS4ML)
- **Optimized Timing**: 8.691 ns
- **Target Device**: xcvu13p-flga2577-2-e
- **Clock Period**: 10.00 ns

## Utilization Estimates

- **BRAM**: 118 (2% of available resources)
- **DSP**: 8 (~0% of available resources)
- **Flip-Flops (FF)**: 18,576 (~0% of available resources)
- **LUTs**: 80,402 (4% of available resources)
- **URAM**: 0

## Results and Analysis

### Cosimulation Reports

- **Optimized Cosimulation Report**:
  - Minimum, average, and maximum latency: 1,249,664 clock cycles [Source: solution1_braintumer_cosim_6_try.html].
  
- **Unoptimized Cosimulation Report**:
  - Minimum, average, and maximum latency: 2,512,289 clock cycles [Source: solution1_braintumer_cosim_unoptimized.html].

### Synthesis Reports

- **Optimized Synthesis Report**: Detailed utilization and performance estimates available in `solution1_braintumer_csynth_6_try.html`.
- **Unoptimized Synthesis Report**: Detailed utilization and performance estimates available in `solution1_braintumer_csynth_unoptimized.html`.

## How to Run the Project

1. **Setup Vivado HLS Environment**: Install Vivado HLS and set up the environment on your local machine.
2. **Clone Repository**:
    ```bash
    git clone https://github.com/yourusername/Brain-Tumor-Detection-HLS-Optimization.git
    ```
3. **Navigate to Project Directory**:
    ```bash
    cd Brain-Tumor-Detection-HLS-Optimization
    ```
4. **Run HLS4ML Optimization**:
    - Follow the instructions in the `hls4ml_optimization.md` file.
5. **Run Vivado HLS Optimization**:
    - Execute the Vivado project using the provided scripts.

## Issues and Challenges

1. **High Latency in Initial Models**: The initial model's latency was very high, ranging up to 3,458,344 cycles, which required significant optimization.
2. **Resource Utilization**: Efficient utilization of BRAM and DSP resources was necessary to meet hardware constraints while optimizing latency.
3. **Clock Constraints**: Ensuring timing constraints were met with a targeted clock period of 10 ns, which required careful pipelining and optimization.

## Future Work

- Further optimization of the pipeline to reduce latency below 1 million cycles.
- Exploration of additional HLS directives and optimizations to enhance hardware resource utilization.
- Integration of more complex models for improved accuracy and robustness.

## Contributions

- **Dhruvkumar Kakadiya**: Model implementation, HLS4ML optimization, Vivado manual optimization, performance analysis.
