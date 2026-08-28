// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "ANEPowerMonitor",
    targets: [
        .target(name: "CAsitop"),
        .executableTarget(
            name: "ANEPowerMonitor",
            dependencies: ["CAsitop"],
            linkerSettings: [
                .linkedLibrary("IOReport"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("IOKit")
            ]
        )
    ]
)
