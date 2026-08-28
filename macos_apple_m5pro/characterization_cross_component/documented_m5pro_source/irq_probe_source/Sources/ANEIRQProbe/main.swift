import Foundation
import CIOReport

let arguments = CommandLine.arguments
guard arguments.count == 6 else {
    fputs("usage: ANEIRQProbe <group> <subgroup|-> <interval_ms> <samples> <output.csv>\n", stderr)
    exit(2)
}

let group = arguments[1]
let subgroup = arguments[2] == "-" ? nil : arguments[2]
let intervalMilliseconds = Double(arguments[3]) ?? 200
let sampleCount = Int(arguments[4]) ?? 50
let outputPath = arguments[5]

guard let channels = IOReportCopyChannelsInGroup(
    group as CFString,
    subgroup.map { $0 as CFString },
    0, 0, 0
) else {
    fputs("no channels for group=\(group) subgroup=\(subgroup ?? "-")\n", stderr)
    exit(3)
}

var subscribedChannels: Unmanaged<CFMutableDictionary>?
guard let subscription = IOReportCreateSubscription(
    nil,
    channels.takeRetainedValue(),
    &subscribedChannels,
    0,
    nil
), let selectedChannels = subscribedChannels?.takeUnretainedValue() else {
    fputs("subscription failed\n", stderr)
    exit(4)
}

FileManager.default.createFile(atPath: outputPath, contents: nil)
guard let output = FileHandle(forWritingAtPath: outputPath) else {
    fputs("cannot open output\n", stderr)
    exit(5)
}
defer { output.closeFile() }

output.write(Data("timestamp,group,subgroup,channel,value\n".utf8))

for _ in 0..<sampleCount {
    autoreleasepool {
        guard let first = IOReportCreateSamples(subscription, selectedChannels, nil) else { return }
        Thread.sleep(forTimeInterval: intervalMilliseconds / 1000)
        guard let second = IOReportCreateSamples(subscription, selectedChannels, nil),
              let delta = IOReportCreateSamplesDelta(
                first.takeUnretainedValue(),
                second.takeUnretainedValue(),
                nil
              ) else { return }
        let timestamp = Date().timeIntervalSince1970
        IOReportIterate(delta.takeRetainedValue()) { channel in
            guard let channel, IOReportChannelGetFormat(channel) == 1 else { return 0 }
            let channelGroup = IOReportChannelGetGroup(channel).map { $0.takeUnretainedValue() as String } ?? ""
            let channelSubgroup = IOReportChannelGetSubGroup(channel).map { $0.takeUnretainedValue() as String } ?? ""
            let channelName = IOReportChannelGetChannelName(channel).map { $0.takeUnretainedValue() as String } ?? ""
            let value = IOReportSimpleGetIntegerValue(channel, 0)
            let row = String(format: "%.6f,%@,%@,%@,%ld\n", timestamp, channelGroup, channelSubgroup, channelName, value)
            output.write(Data(row.utf8))
            return 0
        }
    }
}
