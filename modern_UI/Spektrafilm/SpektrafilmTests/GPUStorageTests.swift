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

}
