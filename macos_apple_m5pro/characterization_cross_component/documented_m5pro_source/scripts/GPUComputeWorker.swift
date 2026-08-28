import Darwin
import Foundation
import Metal
import MetalPerformanceShaders

private var keepRunning = true

private func argument(_ name: String, default defaultValue: String) -> String {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count else {
        return defaultValue
    }
    return CommandLine.arguments[index + 1]
}

signal(SIGTERM) { _ in keepRunning = false }
signal(SIGINT) { _ in keepRunning = false }
setbuf(stdout, nil)

let dutyPercent = min(100.0, max(0.0, Double(argument("--duty-cycle", default: "100")) ?? 100.0))
let matrixSize = max(128, Int(argument("--matrix-size", default: "1024")) ?? 1024)
let cycleSeconds = max(0.05, (Double(argument("--cycle-ms", default: "200")) ?? 200.0) / 1000.0)
let dutyFraction = dutyPercent / 100.0

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal GPU is unavailable")
}

let rowBytes = matrixSize * MemoryLayout<Float>.stride
let byteCount = rowBytes * matrixSize
guard
    let leftBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
    let rightBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
    let resultBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)
else {
    fatalError("Unable to allocate Metal buffers")
}

let leftValues = leftBuffer.contents().bindMemory(to: Float.self, capacity: matrixSize * matrixSize)
let rightValues = rightBuffer.contents().bindMemory(to: Float.self, capacity: matrixSize * matrixSize)
for index in 0..<(matrixSize * matrixSize) {
    leftValues[index] = Float(index % 251) / 251.0
    rightValues[index] = Float((index * 7) % 257) / 257.0
}

let descriptor = MPSMatrixDescriptor(
    rows: matrixSize,
    columns: matrixSize,
    rowBytes: rowBytes,
    dataType: .float32
)
let leftMatrix = MPSMatrix(buffer: leftBuffer, descriptor: descriptor)
let rightMatrix = MPSMatrix(buffer: rightBuffer, descriptor: descriptor)
let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: descriptor)
let multiplication = MPSMatrixMultiplication(
    device: device,
    transposeLeft: false,
    transposeRight: false,
    resultRows: matrixSize,
    resultColumns: matrixSize,
    interiorColumns: matrixSize,
    alpha: 1.0,
    beta: 0.0
)

print("GPU_READY kernel=MPSMatrixMultiplication device=\(device.name) duty=\(dutyPercent)% size=\(matrixSize)")

while keepRunning {
    let cycleStart = ProcessInfo.processInfo.systemUptime
    let activeUntil = cycleStart + cycleSeconds * dutyFraction
    repeat {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            fatalError("Unable to create Metal command buffer")
        }
        multiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: leftMatrix,
            rightMatrix: rightMatrix,
            resultMatrix: resultMatrix
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            fatalError(commandBuffer.error?.localizedDescription ?? "Metal command failed")
        }
    } while keepRunning && ProcessInfo.processInfo.systemUptime < activeUntil
    let remaining = cycleSeconds - (ProcessInfo.processInfo.systemUptime - cycleStart)
    if remaining > 0 {
        Thread.sleep(forTimeInterval: remaining)
    }
}

