"""LPR - License Plate Recognition daemon."""

__version__ = "0.2.0"


def _ort_providers(device: str = "cpu") -> list[str]:
    """Return ONNX Runtime providers for the given device.

    Probes for TensorRT availability and only includes it when libnvinfer
    is actually loadable.  Without this check onnxruntime logs noisy
    errors before falling back to CUDA.
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]

    providers = []

    try:
        import ctypes

        ctypes.CDLL("libnvinfer.so.10")
        providers.append("TensorrtExecutionProvider")
    except OSError:
        pass

    providers += ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers
