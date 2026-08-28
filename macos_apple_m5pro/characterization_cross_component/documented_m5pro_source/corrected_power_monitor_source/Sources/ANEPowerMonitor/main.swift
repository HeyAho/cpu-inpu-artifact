import Foundation
import CAsitop

signal(SIGTERM) { _ in print("SIGTERM"); exit(0) }

print("Init power monitor + GPU freq...")

// ---- GPU DVFM table ----
var gpu_dvfm: [Float] = []
func readDvfm() {
    var iter = io_iterator_t()
    var port: mach_port_t
    if #available(macOS 12, *) { port = kIOMainPortDefault }
    else { port = kIOMasterPortDefault }
    guard let svc = IOServiceMatching("AppleARMIODevice") else { return }
    guard IOServiceGetMatchingServices(port, svc, &iter) == kIOReturnSuccess else { return }
    while case let e = IOIteratorNext(iter), e != IO_OBJECT_NULL {
        var p: Unmanaged<CFMutableDictionary>? = nil
        guard IORegistryEntryCreateCFProperties(e, &p, kCFAllocatorDefault, 0) == kIOReturnSuccess,
              let d = p?.takeRetainedValue() as? [String: AnyObject],
              let data = d["voltage-states9"] else { continue }
        let bytes = data.bytes!.assumingMemoryBound(to: UInt8.self)
        for ii in stride(from: 4, to: (data.length ?? 0) + 4, by: 8) {
            let s = String(format: "0x%02x%02x%02x%02x", bytes[ii-1], bytes[ii-2], bytes[ii-3], bytes[ii-4])
            if let f = Float(s), f > 0 { gpu_dvfm.append(f * 1e-6) }
        }
        IOObjectRelease(e); break
    }
    IOObjectRelease(iter)
    if gpu_dvfm.isEmpty { gpu_dvfm = [612, 912, 1210, 1398] }
    let mhzStr = gpu_dvfm.map { String(format: "%.0f", $0) }.joined(separator: ", ")
    print("GPU DVFM: \(mhzStr) MHz")
}
readDvfm()

// ---- Subscriptions ----
let echn = IOReportCopyChannelsInGroup("Energy Model" as CFString, nil, 0, 0, 0)
let pchn = IOReportCopyChannelsInGroup("PMP" as CFString, nil, 0, 0, 0)
if let e = echn, let p = pchn { IOReportMergeChannels(e.takeUnretainedValue(), p.takeUnretainedValue(), nil) }
guard let pwrCh = echn else { print("no energy"); exit(1) }
var pwrSC: Unmanaged<CFMutableDictionary>? = nil
guard let pwrSub = IOReportCreateSubscription(nil, pwrCh.takeRetainedValue(), &pwrSC, 0, nil) else { print("no pwr sub"); exit(1) }
guard let pwrSCh = pwrSC?.takeUnretainedValue() else { exit(1) }

let gchn = IOReportCopyChannelsInGroup("GPU Stats" as CFString, nil, 0, 0, 0)
var gpuSC: Unmanaged<CFMutableDictionary>? = nil
var gpuSub: IOReportSubscriptionRef?
var gpuSCh: CFMutableDictionary?
if let g = gchn {
    gpuSub = IOReportCreateSubscription(nil, g.takeRetainedValue(), &gpuSC, 0, nil)
    gpuSCh = gpuSC?.takeUnretainedValue()
    print("GPU sub: \(gpuSub != nil ? "OK" : "FAIL")")
}

let interval_ms: Double = 2.0

// ---- CSV ----
var csvName = "hardware_features.csv"
if CommandLine.arguments.count > 1 { csvName = CommandLine.arguments[1]; if !csvName.hasSuffix(".csv") { csvName += ".csv" } }
let csvURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(csvName)
FileManager.default.createFile(atPath: csvURL.path, contents: nil, attributes: nil)
guard let fh = try? FileHandle(forWritingTo: csvURL) else { print("no csv"); exit(1) }
var hdrDone = false; var hdrKeys: [String] = []

let PSTATE = "P"; let VSTATE = "V"

print("Ready: \(csvName) @ \(interval_ms)ms")

// ---- Main loop ----
// CRITICAL: GPU samples must be taken and processed BEFORE any power sample call.
// Calling IOReportCreateSamples on the Energy subscription corrupts GPU state counters.
while true {
    autoreleasepool {
        // ==========================================
        // PHASE 1: GPU sampling + processing (MUST be first!)
        // ==========================================
        var gpuFreq: Float = 0
        var gpuClusters: Int = 0

        if let gSub = gpuSub, let gSCh = gpuSCh {
            let gA = IOReportCreateSamples(gSub, gSCh, nil)
            Thread.sleep(forTimeInterval: interval_ms / 1000.0)
            let gB = IOReportCreateSamples(gSub, gSCh, nil)

            if let gD = IOReportCreateSamplesDelta(gA?.takeUnretainedValue(), gB?.takeUnretainedValue(), nil) {
                IOReportIterate(gD.takeRetainedValue()) { sample in
                    guard let sd = sample else { return Int32(0) }
                    let sg = IOReportChannelGetSubGroup(sd)?.takeUnretainedValue() as? String ?? ""
                    let cn = IOReportChannelGetChannelName(sd)?.takeUnretainedValue() as? String ?? ""
                    if sg == "GPU Performance States" && cn == "GPUPH" {
                        var cf: Float = 0; var ca: Float = 0
                        for i in 0..<Int(IOReportStateGetCount(sd)) {
                            let nm = IOReportStateGetNameForIndex(sd, Int32(i))?.takeUnretainedValue() as? String ?? ""
                            let rs = IOReportStateGetResidency(sd, Int32(i))
                            if nm.contains(PSTATE) || nm.contains(VSTATE) {
                                let di = i - 1
                                if di >= 0 && di < gpu_dvfm.count { cf += Float(rs) * gpu_dvfm[di] }
                                ca += Float(rs)
                            }
                        }
                        if ca > 0 { gpuFreq += cf / ca; gpuClusters += 1 }
                    }
                    return Int32(0)
                }
            }
        }
        if gpuClusters > 0 { gpuFreq /= Float(gpuClusters) }

        // ==========================================
        // PHASE 2: Power sampling + processing (after GPU is fully done)
        // ==========================================
        var rawPower: [String: Float] = [:]

        let pA = IOReportCreateSamples(pwrSub, pwrSCh, nil)
        Thread.sleep(forTimeInterval: interval_ms / 1000.0)
        let pB = IOReportCreateSamples(pwrSub, pwrSCh, nil)

        let pD = IOReportCreateSamplesDelta(pA?.takeUnretainedValue(), pB?.takeUnretainedValue(), nil)
        IOReportIterate(pD?.takeRetainedValue()) { sample in
            guard let sd = sample else { return Int32(0) }
            if IOReportChannelGetFormat(sd) == 1 {
                if let cn = IOReportChannelGetChannelName(sd), let gn = IOReportChannelGetGroup(sd) {
                    let c = cn.takeUnretainedValue() as String
                    let g = gn.takeUnretainedValue() as String
                    if g == "Energy Model" || g == "PMP" {
                        switch c {
                        case "ANE", "GPU", "GPU Energy", "PCPU", "MCPU0", "MCPU1", "CPU Energy":
                            rawPower[c, default: 0] += Float(IOReportSimpleGetIntegerValue(sd, 0)) / 1000.0 / Float(interval_ms / 1000.0)
                        default:
                            break
                        }
                    }
                }
            }
            return Int32(0)
        }

        // ---- Write CSV ----
        let ts = Date().timeIntervalSince1970
        let superPower = rawPower["PCPU"] ?? 0
        let performance0 = rawPower["MCPU0"] ?? 0
        let performance1 = rawPower["MCPU1"] ?? 0
        let performancePower = performance0 + performance1
        let measuredTotal = rawPower["CPU Energy"] ?? 0
        let cpuTotal = measuredTotal > 0 ? measuredTotal : superPower + performancePower
        let powerDict: [String: Float] = [
            "ANE": rawPower["ANE"] ?? 0,
            "GPU": rawPower["GPU"] ?? 0,
            "GPU_Energy": rawPower["GPU Energy"] ?? 0,
            "GPU_Freq": gpuFreq,
            "CPU_Super": superPower,
            "CPU_Performance_0": performance0,
            "CPU_Performance_1": performance1,
            "CPU_Performance": performancePower,
            "CPU_Total": cpuTotal,
        ]

        if !hdrDone {
            hdrKeys = powerDict.keys.sorted()
            let hr = "Timestamp," + hdrKeys.joined(separator: ",") + "\n"
            if let d = hr.data(using: .utf8) { fh.write(d) }
            hdrDone = true
        }
        var row = [String(format: "%.6f", ts)]
        for k in hdrKeys { row.append(String(format: "%.4f", powerDict[k] ?? 0)) }
        if let d = (row.joined(separator: ",") + "\n").data(using: .utf8) { fh.write(d) }
    }
}
