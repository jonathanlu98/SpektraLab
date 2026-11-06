//
//  GPUStorage.swift — where a texture the CPU also touches lives.
//
//  On Apple silicon the answer is one set of pages, `.shared`. On an Intel Mac
//  it is not: Metal refuses a shared texture (`MTLStorageModeShared not allowed
//  for textures`, under the validation layer; without it the descriptor is
//  accepted and the behaviour is undefined). The legal form is `.managed`, two
//  copies with an explicit hand-over. `replace(region:)` carries the CPU's copy
//  to the GPU by itself; the way back is `synchronizeForCPU()`.
//
//  A texture over an `MTLBuffer` stays `.shared` everywhere, which is why the
//  engine's zero-copy result (`MetalGpu::texture`) needs none of this.
//
import Metal

extension MTLDevice {
    /// Asked of the GPU rather than decided by the build: an arm64 slice can
    /// only meet an Apple GPU, but an x86_64 one meets both under Rosetta.
    var cpuVisibleTextureStorage: MTLStorageMode {
        supportsFamily(.apple1) ? .shared : .managed
    }
}

extension MTLTexture {
    /// Make what the GPU has written readable by `getBytes`, and wait for it.
    /// Nothing to do unless the texture is `.managed`.
    ///
    /// A whole-texture copy across the bus, so it belongs at the door of a
    /// read and never in the draw loop.
    func synchronizeForCPU() {
        guard storageMode == .managed,
              let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return }
        blit.synchronize(resource: self)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
