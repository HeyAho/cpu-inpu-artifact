// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "ANEIRQProbe",
    targets: [
        .target(name: "CIOReport"),
        .executableTarget(
            name: "ANEIRQProbe",
            dependencies: ["CIOReport"],
            linkerSettings: [
                .linkedLibrary("IOReport"),
                .linkedFramework("CoreFoundation")
            ]
        )
    ]
)
