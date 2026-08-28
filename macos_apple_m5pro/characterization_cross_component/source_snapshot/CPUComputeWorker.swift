import Accelerate
import Darwin
import Foundation

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
let workerCount = max(1, Int(argument("--workers", default: "1")) ?? 1)
let matrixSize = max(64, Int(argument("--matrix-size", default: "512")) ?? 512)
let cycleSeconds = max(0.05, (Double(argument("--cycle-ms", default: "200")) ?? 200.0) / 1000.0)
let dutyFraction = dutyPercent / 100.0

print("CPU_READY kernel=Accelerate_cblas_sgemm duty=\(dutyPercent)% workers=\(workerCount) size=\(matrixSize)")

let group = DispatchGroup()
for workerIndex in 0..<workerCount {
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        let elementCount = matrixSize * matrixSize
        var left = [Float](repeating: 0, count: elementCount)
        var right = [Float](repeating: 0, count: elementCount)
        var result = [Float](repeating: 0, count: elementCount)
        for index in 0..<elementCount {
            left[index] = Float((index + workerIndex * 17) % 251) / 251.0
            right[index] = Float((index * 7 + workerIndex * 29) % 257) / 257.0
        }
        while keepRunning {
            let cycleStart = ProcessInfo.processInfo.systemUptime
            let activeUntil = cycleStart + cycleSeconds * dutyFraction
            repeat {
                left[0] += 0.000_001
                left.withUnsafeBufferPointer { leftBuffer in
                    right.withUnsafeBufferPointer { rightBuffer in
                        result.withUnsafeMutableBufferPointer { resultBuffer in
                            cblas_sgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(matrixSize), Int32(matrixSize), Int32(matrixSize),
                                1.0, leftBuffer.baseAddress!, Int32(matrixSize),
                                rightBuffer.baseAddress!, Int32(matrixSize),
                                0.0, resultBuffer.baseAddress!, Int32(matrixSize)
                            )
                        }
                    }
                }
            } while keepRunning && ProcessInfo.processInfo.systemUptime < activeUntil
            let remaining = cycleSeconds - (ProcessInfo.processInfo.systemUptime - cycleStart)
            if remaining > 0 {
                Thread.sleep(forTimeInterval: remaining)
            }
        }
        if result[0].isNaN {
            fputs("unexpected NaN\n", stderr)
        }
        group.leave()
    }
}
group.wait()

