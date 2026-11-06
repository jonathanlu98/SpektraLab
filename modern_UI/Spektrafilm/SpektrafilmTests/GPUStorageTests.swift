//
//  GPUStorageTests.swift — a texture the CPU touches, on the GPU this runs on.
//
//  AGENTS.md trap 36. Seen red on an RX 6800 by removing `synchronizeForCPU()`
//  from `rgba16Bytes`: the bytes read all zeros, with no error from anything.
//
import Metal
import XCTest

@MainActor
final class GPUStorageTests: XCTestCase {
    private func device() throws -> MTLDevice { try XCTUnwrap(MTLCreateSystemDefaultDevice()) }

    /// Clear a texture on the GPU, which is the plainest write there is: the
    /// CPU's copy of a `.managed` texture knows nothing about it.
    private func clear(_ texture: MTLTexture, to value: Double, device: MTLDevice) throws {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: value, green: value, blue: value, alpha: 1)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        try XCTUnwrap(cb.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    func testTheStorageIsOneThisGPUAllows() throws {
        let gpu = try device()
        XCTAssertEqual(gpu.cpuVisibleTextureStorage, gpu.supportsFamily(.apple1) ? .shared : .managed)
    }

    func testTheWritableTextureHandsTheGPUsWriteToTheCPU() throws {
        let gpu = try device()
        let store = TextureStore(device: gpu, arena: MemoryArena())
        let texture = try XCTUnwrap(store.makeWritable(width: 4, height: 2))
        try clear(texture, to: 0.5, device: gpu)

        let bytes = texture.rgba16Bytes()
        XCTAssertEqual(bytes.count, 4 * 2 * 8)
        let first = bytes.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self).prefix(4)) }
        XCTAssertEqual(Double(first[0]) / 65535, 0.5, accuracy: 0.001,
                       "the read saw what the CPU's copy held rather than what the GPU wrote")
        XCTAssertEqual(first[3], 65535)
        XCTAssertNotNil(texture.makeCGImage())
    }

    func testASecondWriteIsSeenByASecondRead() throws {
        // The hand-over is per write, not once: a texture that is synchronised
        // on first read and trusted afterwards would pass the test above.
        let gpu = try device()
        let store = TextureStore(device: gpu, arena: MemoryArena())
        let texture = try XCTUnwrap(store.makeWritable(width: 2, height: 2))
        for value in [0.25, 0.75] {
            try clear(texture, to: value, device: gpu)
            let bytes = texture.rgba16Bytes()
            let red = bytes.withUnsafeBytes { $0.bindMemory(to: UInt16.self)[0] }
            XCTAssertEqual(Double(red) / 65535, value, accuracy: 0.001)
        }
    }

    func testAnUploadReadsBackWithNoHandOverAtAll() throws {
        // `replace(region:)` is the CPU's own write, so `getBytes` needs nothing.
        let gpu = try device()
        let store = TextureStore(device: gpu, arena: MemoryArena())
        var px = [UInt16](repeating: 0, count: 4)
        px[0] = 1234
        let data = px.withUnsafeBytes { Data($0) }
        let texture = try XCTUnwrap(store.uploadRGBA16(data: data, width: 1, height: 1))
        var read = [UInt16](repeating: 0, count: 4)
        texture.getBytes(&read, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0)
        XCTAssertEqual(read[0], 1234)
    }

    #if arch(x86_64)
    /// Hand-derived IEEE 754 binary16, because the point of this function is
    /// that `Float16` is not there to compare it with on this architecture.
    func testHalfFloatBitsAreIEEEBinary16() {
        let cases: [(Float, UInt16)] = [
            (0, 0x0000), (1, 0x3C00), (0.5, 0x3800), (-2, 0xC000), (1.0 / 3.0, 0x3555),
            (65504, 0x7BFF),                     // the largest finite value
            (1e6, 0x7C00),                       // overflows to infinity
            (6.103515625e-05, 0x0400),           // the smallest normal, 2^-14
            (5.960464477539063e-08, 0x0001),     // the smallest subnormal, 2^-24
            (1 + 0x1p-11, 0x3C00),               // a tie, to even
            (1 + 3 * 0x1p-11, 0x3C02),           // a tie, to even, upward
        ]
        let got = CineonLUT.halfFloatBits(cases.map(\.0))
        for (i, c) in cases.enumerated() {
            XCTAssertEqual(got[i], c.1, "\(c.0)")
        }
    }
    #endif
}
