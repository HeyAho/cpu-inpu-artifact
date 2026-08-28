import Foundation
import CAsitop

let intervalMs = 100.0
let energy = IOReportCopyChannelsInGroup("Energy Model" as CFString, nil, 0, 0, 0)
let pmp = IOReportCopyChannelsInGroup("PMP" as CFString, nil, 0, 0, 0)
if let energy, let pmp {
    IOReportMergeChannels(energy.takeUnretainedValue(), pmp.takeUnretainedValue(), nil)
}
guard let channels = energy else {
    fputs("Energy Model channels unavailable\n", stderr)
    exit(1)
}

var subscribed: Unmanaged<CFMutableDictionary>?
guard let subscription = IOReportCreateSubscription(
    nil,
    channels.takeRetainedValue(),
    &subscribed,
    0,
    nil
), let selected = subscribed?.takeUnretainedValue() else {
    fputs("Energy Model subscription failed\n", stderr)
    exit(2)
}

guard let first = IOReportCreateSamples(subscription, selected, nil) else {
    exit(3)
}
Thread.sleep(forTimeInterval: intervalMs / 1000.0)
guard let second = IOReportCreateSamples(subscription, selected, nil),
      let delta = IOReportCreateSamplesDelta(
        first.takeUnretainedValue(),
        second.takeUnretainedValue(),
        nil
      ) else {
    exit(4)
}

print("group,subgroup,channel,raw_delta,reported_power")
IOReportIterate(delta.takeRetainedValue()) { sample in
    guard let sample, IOReportChannelGetFormat(sample) == 1 else { return 0 }
    let group = IOReportChannelGetGroup(sample)?.takeUnretainedValue() as? String ?? ""
    guard group == "Energy Model" || group == "PMP" else { return 0 }
    let subgroup = IOReportChannelGetSubGroup(sample)?.takeUnretainedValue() as? String ?? ""
    let channel = IOReportChannelGetChannelName(sample)?.takeUnretainedValue() as? String ?? ""
    let raw = IOReportSimpleGetIntegerValue(sample, 0)
    let power = Double(raw) / 1000.0 / (intervalMs / 1000.0)
    print("\(group),\(subgroup),\(channel),\(raw),\(String(format: "%.6f", power))")
    return 0
}
